import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import copy
import random
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.hparams import HParams
from utils.metrics import DirectionalQuantileLoss, DilateLoss, DilateQuantileLoss, normalized_quantile_loss, mean_directional_accuracy
from utils.models import SparseTemporalFusionTransformer
from utils.visualize import visualize
from preprocessing import preprocess

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import PoissonLoss, QuantileLoss, SMAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from matplotlib import pyplot as plt
import matplotlib

warnings.filterwarnings("ignore")

# setting for print out korean in figures 
plt.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# hyperparameter - using argparse and parameter module
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='experiment data', default='vol')
parser.add_argument('--model', type=str, help='model name', default='tft')
parser.add_argument('--loss', type=str, help='loss function', default='quantile')
parser.add_argument('--symbol', type=str, help='stock symbol', default=None)
parser.add_argument('--transfer', type=str, help='transfer model data', default=None)
parser.add_argument('--idx', type=int, help='experiment number',  default=None)
parser.add_argument('--ws', type=str, help='machine number', default='9')
parser.add_argument('--gpu_index', '-g', type=int, help='GPU index', default=0)
parser.add_argument('--ngpu', type=int, help='0 = CPU.', default=1)
parser.add_argument('--distributed_backend', type=str, help="'dp' or 'ddp' for multi-gpu training", default=None)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

# GPU allocation
if args.gpu_index:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_index)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        torch.cuda.set_device(device)
        print("Current cuda device", torch.cuda.current_device())

hparam_file = os.path.join(os.getcwd(), "hparams.yaml")
config = HParams.load(hparam_file)
asset_root = config.asset_root[args.ws][args.model]
asset_path = os.path.join(asset_root, args.data)
optuna_path = os.path.join(asset_path, 'optuna_model')

# seed
if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

# preprocessing
data = preprocess(args.data, args.symbol)

# Training setting
max_prediction_length = config.experiment['max_prediction_length']
max_encoder_length = config.experiment['max_encoder_length'][args.data]
train_boundary = config.experiment['train_boundary'][args.data]
valid_boundary = config.experiment['valid_boundary'][args.data]
test_boundary = config.experiment['test_boundary'][args.data]

training = TimeSeriesDataSet(
    data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(train_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime(valid_boundary))],
    time_idx=config.dataset_setting[args.data]['time_idx'],
    target=config.dataset_setting[args.data]['target'],
    group_ids=config.dataset_setting[args.data]['group_ids'],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    # min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=config.dataset_setting[args.data]['static_categoricals'],
    static_reals=config.dataset_setting[args.data]['static_reals'],
    time_varying_known_categoricals=config.dataset_setting[args.data]['time_varying_known_categoricals'],
    variable_groups=config.dataset_setting[args.data]['variable_groups'],  # group of categorical variables can be treated as one variable
    time_varying_known_reals=config.dataset_setting[args.data]['time_varying_known_reals'],
    time_varying_unknown_categoricals=config.dataset_setting[args.data]['time_varying_unknown_categoricals'],
    time_varying_unknown_reals=config.dataset_setting[args.data]['time_varying_unknown_reals'],
    target_normalizer=GroupNormalizer(groups=config.dataset_setting[args.data]['group_ids']),  # normalize by group
    allow_missings=True, # allow time_idx missing
    scalers={StandardScaler(): config.dataset_setting[args.data]['time_varying_unknown_reals']},
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(valid_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime(test_boundary))], predict=True, stop_randomization=True)
if args.data == 'vol':
    test = TimeSeriesDataSet.from_dataset(training, data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(test_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime('2019.06.29'))], predict=True, stop_randomization=True)
else:
    test = TimeSeriesDataSet.from_dataset(training, data[lambda x: pd.to_datetime(x.date) >= pd.to_datetime(test_boundary)], predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = config.experiment['batch_size']  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# hyperparameter tuning
if not os.path.exists(optuna_path):
    os.makedirs(optuna_path)

study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path=optuna_path,
    log_dir=asset_path,
    n_trials=100,
    timeout=None,
    max_epochs=20, #50
    gradient_clip_val_range=(0.01, 1.0), #(0.01, 1.0)
    hidden_size_range=(8, 160), #(8, 128)
    hidden_continuous_size_range=(8, 128), #(8, 128)
    attention_head_size_range=(1, 4), #(1, 4)
    learning_rate_range=(0.001, 0.1), #(0.001, 0.1)
    dropout_range=(0.1, 0.5),
    # trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=1000,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    verbose=1,
)

# save study results - also we can resume tuning at a later point in time
with open(os.path.join(optuna_path, "test_study.pkl"), "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)

# test on the best model
for ckpt_file in os.listdir(os.path.join(optuna_path, "trial_"+str(study.best_trial.number))):
    if ckpt_file.endswith(".ckpt"):
        best_model_path = os.path.join(os.path.join(optuna_path, "trial_"+str(study.best_trial.number)), ckpt_file)
print(best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calcualte quantile loss on test set
best_tft.to(torch.device('cpu'))
actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
raw_predictions = best_tft.predict(test_dataloader, mode='raw')
raw_predictions = raw_predictions['prediction']
normalized_loss = normalized_quantile_loss(actuals, raw_predictions)
print(f'Normalized quantile loss - p10: {normalized_loss[0]}, p50: {normalized_loss[1]}, p90: {normalized_loss[2]}')

# calculate mean directional accuracy on test set
mda = mean_directional_accuracy(actuals, raw_predictions)
one_day_mda = mean_directional_accuracy(actuals[:, :2], raw_predictions[:, :2, :])
print(f'MDA: {mda}, MDA-1day: {one_day_mda}')

'''
# print example
{'gradient_clip_val': 0.3498626806469764, 'hidden_size': 78, 'dropout': 0.11547495781892213, 
'hidden_continuous_size': 26, 'attention_head_size': 3, 'learning_rate': 0.06901691924557539}                                                                                                        
'''






