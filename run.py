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
from preprocessing import preprocess
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer

from pytorch_forecasting.metrics import PoissonLoss, QuantileLoss, SMAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

warnings.filterwarnings("ignore")

# hyperparameter - using argparse and parameter module
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='experiment data', default='volatility')
parser.add_argument('--idx', type=int, help='experiment number',  default=0)
parser.add_argument('--ws', type=str, help='machine number', default='9')
parser.add_argument('--gpu_index', '-g', type=int, default="0", help='GPU index')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

device = torch.device("cuda:%d" % args.gpu_index if torch.cuda.is_available() else "cpu")
hparam_file = os.path.join(os.getcwd(), "hparams.yaml")

config = HParams.load(hparam_file)
asset_root = config.asset_root[args.ws]

# seed
if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# preprocessing
data = preprocess(args.data)

# Training setting
max_prediction_length = config.experiment['max_prediction_length']
max_encoder_length = config.experiment['max_encoder_length']
valid_boundary = config.experiment['valid_boundary']
test_boundary = config.experiment['test_boundary']

training = TimeSeriesDataSet(
    data[lambda x: x.date < valid_boundary],
    time_idx=config.dataset_setting[args.data]['time_idx'],
    target=config.dataset_setting[args.data]['target'],
    group_ids=config.dataset_setting[args.data]['group_ids'],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
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
validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: (x.date >= valid_boundary) & (x.date < test_boundary)], predict=True, stop_randomization=True)
test = TimeSeriesDataSet.from_dataset(training, data[lambda x: x.date >= test_boundary], predict=True, stop_randomization=True)
# test = TimeSeriesDataSet.from_dataset(training, data[lambda x: (x.date >= test_boundary) & (x.date < '2019.06.29')], predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = config.experiment['batch_size']  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, y in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
print('baseline MAE: ', (actuals - baseline_predictions).abs().mean().item())

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=20, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate

if not os.path.exists(asset_root):
    os.makedirs(asset_root)
logger = TensorBoardLogger(save_dir=asset_root, name=args.data, version=args.idx)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=config.experiment['epoch'],
    gpus=args.ngpu,
    weights_summary=config.experiment['weights_summary'],
    gradient_clip_val=config.experiment['gradient_clip'],
    limit_train_batches=config.experiment['limit_train_batches'],  # coment in for training, running valiation every 30 batches
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=config.experiment['lr'], 
    hidden_size=config.model['hidden_size'], 
    attention_head_size=config.model['attention_head_size'],
    dropout=config.model['dropout'],
    hidden_continuous_size=config.model['hidden_continuous_size'],
    output_size=config.model['output_size'],  # 7 quantiles by default
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    log_interval=config.model['log_interval'],  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=config.model['reduce_on_plateau_patience'],
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"best model path: {best_model_path}")
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calcualte mean absolute error on validation set
actuals = torch.cat([y for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
print('validation MAE: ', (actuals - predictions).abs().mean())

# For testing, you should append test_step(), test_epoch_end() method in Basemodel class. (filepath: pytorch-forecasting > models > base_model.py)
# Test
trainer.test(
    best_tft,
    test_dataloaders=test_dataloader, 
    verbose=True,
)


##### Visualizing Part #####
# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
# raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
# for idx in range(10): 
#     best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)

# predictions, x = best_tft.predict(val_dataloader, return_x=True)
# predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
# best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

# interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
# best_tft.plot_interpretation(interpretation)

