import os
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
plt.rcParams["font.family"] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_index)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cuda:0":
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())
hparam_file = os.path.join(os.getcwd(), "hparams.yaml")

config = HParams.load(hparam_file)
asset_root = config.asset_root[args.ws][args.model]

# seed
if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

# preprocessing
data = preprocess(args.data)

# Training setting
max_prediction_length = config.experiment['max_prediction_length']
max_encoder_length = 63 # 252/4

train_boundary = config.experiment['train_boundary'][args.data]
valid_boundary = config.experiment['valid_boundary'][args.data]
test_boundary = config.experiment['test_boundary'][args.data]

training = TimeSeriesDataSet(
    data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(train_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime(valid_boundary))],
    time_idx=config.dataset_setting[args.data]['time_idx'],
    target=config.dataset_setting[args.data]['target'],
    group_ids=config.dataset_setting[args.data]['group_ids'],
#     min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
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
# batch_size = config.experiment['batch_size']  # set this between 32 to 128
batch_size = 512 #128*4
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=10, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate

if not os.path.exists(asset_root):
    os.makedirs(asset_root)

logger = TensorBoardLogger(save_dir=asset_root, name=args.data)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=config.experiment['epoch'], # 50
    gpus=args.ngpu,
    weights_summary=config.experiment['weights_summary'],
    gradient_clip_val=config.experiment['gradient_clip'],
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

if args.loss == 'directional':
    tft_loss = DirectionalQuantileLoss(quantiles=[0.1, 0.5, 0.9], weight=config.model['weight'])
elif args.loss == 'dilate':
    tft_loss = DilateQuantileLoss(quantiles=[0.1, 0.5, 0.9], alpha=config.model['alpha'], weight=config.model['weight'])
else:
    tft_loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

if args.model == 'tft':
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config.experiment['lr'][args.data], 
        hidden_size=config.model['hidden_size'], 
        attention_head_size=config.model['attention_head_size'],
        dropout=config.model['dropout'],
        hidden_continuous_size=config.model['hidden_continuous_size'],
        output_size=config.model['output_size'],  # 7 quantiles by default
        loss=tft_loss,
        log_interval=config.model['log_interval'],  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=config.model['reduce_on_plateau_patience'],
        optimizer=config.model['optimizer'], # Optimizer, "ranger", "adam" or "adamw". Defaults to "ranger".
    )
elif args.model == 'stft':
    tft = SparseTemporalFusionTransformer.from_dataset(
        training,
        alpha=config.model['ent_ratio'],
        learning_rate=config.experiment['lr'][args.data], 
        hidden_size=config.model['hidden_size'], 
        attention_head_size=config.model['attention_head_size'],
        dropout=config.model['dropout'],
        hidden_continuous_size=config.model['hidden_continuous_size'],
        output_size=config.model['output_size'],  # 7 quantiles by default
        loss=tft_loss,
        log_interval=config.model['log_interval'],  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=config.model['reduce_on_plateau_patience'],
        optimizer=config.model['optimizer'], # Optimizer, "ranger", "adam" or "adamw". Defaults to "ranger".
    )

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

# load the best model according to the validation loss
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"best model path: {best_model_path}")

############################### 2nd stage ##########################################
# Training setting
max_prediction_length = config.experiment['max_prediction_length']
max_encoder_length = config.experiment['max_encoder_length'][args.data]

training = TimeSeriesDataSet(
    data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(train_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime(valid_boundary))],
    time_idx=config.dataset_setting[args.data]['time_idx'],
    target=config.dataset_setting[args.data]['target'],
    group_ids=config.dataset_setting[args.data]['group_ids'],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
#     min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
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
batch_size = config.experiment['batch_size']  # 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

lr_logger = LearningRateMonitor()  # log the learning rate

logger = TensorBoardLogger(save_dir=asset_root, name=args.data)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=5,
    gpus=args.ngpu,
    weights_summary=config.experiment['weights_summary'],
    gradient_clip_val=config.experiment['gradient_clip'],
    # callbacks=[lr_logger, early_stop_callback],
    callbacks=[lr_logger],
    logger=logger,
)

if args.model == 'tft':
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config.experiment['lr'][args.data], 
        hidden_size=config.model['hidden_size'], 
        attention_head_size=config.model['attention_head_size'],
        dropout=config.model['dropout'],
        hidden_continuous_size=config.model['hidden_continuous_size'],
        output_size=config.model['output_size'],  # 7 quantiles by default
        loss=tft_loss,
        log_interval=config.model['log_interval'],  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=config.model['reduce_on_plateau_patience'],
        optimizer=config.model['optimizer'], # Optimizer, "ranger", "adam" or "adamw". Defaults to "ranger".
    )
elif args.model == 'stft':
    tft = SparseTemporalFusionTransformer.from_dataset(
        training,
        alpha=config.model['ent_ratio'],
        learning_rate=config.experiment['lr'][args.data], 
        hidden_size=config.model['hidden_size'], 
        attention_head_size=config.model['attention_head_size'],
        dropout=config.model['dropout'],
        hidden_continuous_size=config.model['hidden_continuous_size'],
        output_size=config.model['output_size'],  # 7 quantiles by default
        loss=tft_loss,
        log_interval=config.model['log_interval'],  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=config.model['reduce_on_plateau_patience'],
        optimizer=config.model['optimizer'], # Optimizer, "ranger", "adam" or "adamw". Defaults to "ranger".
    )

ckpt = torch.load(best_model_path)
tft.load_state_dict(ckpt['state_dict'], strict=False)

# fit network
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

# For testing, you should append test_step(), test_epoch_end() method in Basemodel class. (filepath: pytorch-forecasting > models > base_model.py)
# Test
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"best model path: {best_model_path}")
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

trainer.test(
    best_tft,
    test_dataloaders=test_dataloader, 
    verbose=True,
)

##### Visualizing Part #####
best_tft.to(torch.device('cpu'))

image_root = os.path.join(logger.log_dir, 'images')
if not os.path.exists(image_root):
    os.makedirs(image_root)

topk_groups = visualize(training, test_dataloader, best_tft, image_root)
print(f"figure path: {image_root}")

##### Quantitative Analysis Part #####
# calculate quantile loss on test set
actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
raw_predictions = best_tft.predict(test_dataloader, mode='raw')
raw_predictions = raw_predictions['prediction']
normalized_loss = normalized_quantile_loss(actuals, raw_predictions)
print(f'Normalized quantile loss - p10: {normalized_loss[0]}, p50: {normalized_loss[1]}, p90: {normalized_loss[2]}')

# calculate mean directional accuracy on test set
mda = mean_directional_accuracy(actuals, raw_predictions)
one_day_mda = mean_directional_accuracy(actuals[:, :2], raw_predictions[:, :2, :])
print(f'MDA: {mda}, MDA-1day: {one_day_mda}')

# calculate mean directional accuracy on Top-10 set
topk_test = TimeSeriesDataSet.from_dataset(training, data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(test_boundary)) & (x['Symbol'].isin(topk_groups))], predict=True, stop_randomization=True)
topk_test_dataloader = topk_test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
topk_actuals = torch.cat([y[0] for x, y in iter(topk_test_dataloader)])
topk_raw_predictions = best_tft.predict(topk_test_dataloader, mode='raw')
topk_raw_predictions = topk_raw_predictions['prediction']
topk_mda = mean_directional_accuracy(topk_actuals, topk_raw_predictions)
topk_one_day_mda = mean_directional_accuracy(topk_actuals[:, :2], topk_raw_predictions[:, :2, :])
print(f'Top10 Symbols: {topk_groups}')
print(f'(Top10) MDA: {topk_mda}, (Top10) MDA-1day: {topk_one_day_mda}')