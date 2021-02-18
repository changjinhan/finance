import sys
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
from utils.metrics import normalized_quantile_loss
from utils.models import SparseTemporalFusionTransformer
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

# setting for print out korean in figures 
plt.rcParams["font.family"] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")

# hyperparameter
WS = '9'
SEED = 42
GPU_NUM = 0
NGPU = 1
DATA = 'kospi'
TRANSFER = None

# GPU allocation
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_NUM)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cuda:0":
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())
hparam_file = os.path.join(os.getcwd(), "hparams.yaml")
config = HParams.load(hparam_file)
asset_root = config.asset_root[WS]

# preprocessing
data = preprocess(DATA)

# Training setting
max_prediction_length = config.experiment['max_prediction_length']
max_encoder_length = 63 # 252/4

train_boundary = config.experiment['train_boundary'][DATA]
valid_boundary = config.experiment['valid_boundary'][DATA]
test_boundary = config.experiment['test_boundary'][DATA]

training = TimeSeriesDataSet(
    data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(train_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime(valid_boundary))],
    time_idx=config.dataset_setting[DATA]['time_idx'],
    target=config.dataset_setting[DATA]['target'],
    group_ids=config.dataset_setting[DATA]['group_ids'],
#     min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=config.dataset_setting[DATA]['static_categoricals'],
    static_reals=config.dataset_setting[DATA]['static_reals'],
    time_varying_known_categoricals=config.dataset_setting[DATA]['time_varying_known_categoricals'],
    variable_groups=config.dataset_setting[DATA]['variable_groups'],  # group of categorical variables can be treated as one variable
    time_varying_known_reals=config.dataset_setting[DATA]['time_varying_known_reals'],
    time_varying_unknown_categoricals=config.dataset_setting[DATA]['time_varying_unknown_categoricals'],
    time_varying_unknown_reals=config.dataset_setting[DATA]['time_varying_unknown_reals'],
    target_normalizer=GroupNormalizer(groups=config.dataset_setting[DATA]['group_ids']),  # normalize by group
    allow_missings=True, # allow time_idx missing
    scalers={StandardScaler(): config.dataset_setting[DATA]['time_varying_unknown_reals']},
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(valid_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime(test_boundary))], predict=True, stop_randomization=True)
if DATA == 'vol':
    test = TimeSeriesDataSet.from_dataset(training, data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(test_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime('2019.06.29'))], predict=True, stop_randomization=True)
else:
    test = TimeSeriesDataSet.from_dataset(training, data[lambda x: pd.to_datetime(x.date) >= pd.to_datetime(test_boundary)], predict=True, stop_randomization=True)

# create dataloaders for model
# batch_size = config.experiment['batch_size']  # set this between 32 to 128
batch_size = 256 # 64*4
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=10, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate

if not os.path.exists(asset_root):
    os.makedirs(asset_root)

logger = TensorBoardLogger(save_dir=asset_root, name=DATA)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=config.experiment['epoch'], # 50
    gpus=NGPU,
    weights_summary=config.experiment['weights_summary'],
    gradient_clip_val=config.experiment['gradient_clip'],
    limit_train_batches=config.experiment['limit_train_batches'], # 30
#     callbacks=[lr_logger, early_stop_callback],
    callbacks=[lr_logger],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=config.experiment['lr'][DATA], 
    hidden_size=config.model['hidden_size'], 
    attention_head_size=config.model['attention_head_size'],
    dropout=config.model['dropout'],
    hidden_continuous_size=config.model['hidden_continuous_size'],
    output_size=config.model['output_size'],  # 7 quantiles by default
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    log_interval=config.model['log_interval'],  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=config.model['reduce_on_plateau_patience'], # 4
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
# best_model_path = "/data3/finance/assets/cjhan/tft_logs/kospi/version_220/checkpoints/epoch=47.ckpt"

############################### 2nd stage ##########################################
# Training setting
max_prediction_length = config.experiment['max_prediction_length']
max_encoder_length = config.experiment['max_encoder_length'][DATA]

training = TimeSeriesDataSet(
    data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(train_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime(valid_boundary))],
    time_idx=config.dataset_setting[DATA]['time_idx'],
    target=config.dataset_setting[DATA]['target'],
    group_ids=config.dataset_setting[DATA]['group_ids'],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
#     min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=config.dataset_setting[DATA]['static_categoricals'],
    static_reals=config.dataset_setting[DATA]['static_reals'],
    time_varying_known_categoricals=config.dataset_setting[DATA]['time_varying_known_categoricals'],
    variable_groups=config.dataset_setting[DATA]['variable_groups'],  # group of categorical variables can be treated as one variable
    time_varying_known_reals=config.dataset_setting[DATA]['time_varying_known_reals'],
    time_varying_unknown_categoricals=config.dataset_setting[DATA]['time_varying_unknown_categoricals'],
    time_varying_unknown_reals=config.dataset_setting[DATA]['time_varying_unknown_reals'],
    target_normalizer=GroupNormalizer(groups=config.dataset_setting[DATA]['group_ids']),  # normalize by group
    allow_missings=True, # allow time_idx missing
    scalers={StandardScaler(): config.dataset_setting[DATA]['time_varying_unknown_reals']},
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(valid_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime(test_boundary))], predict=True, stop_randomization=True)
if DATA == 'vol':
    test = TimeSeriesDataSet.from_dataset(training, data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(test_boundary)) & (pd.to_datetime(x.date) < pd.to_datetime('2019.06.29'))], predict=True, stop_randomization=True)
else:
    test = TimeSeriesDataSet.from_dataset(training, data[lambda x: pd.to_datetime(x.date) >= pd.to_datetime(test_boundary)], predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = config.experiment['batch_size']  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

lr_logger = LearningRateMonitor()  # log the learning rate

logger = TensorBoardLogger(save_dir=asset_root, name=DATA)  # logging results to a tensorboard

trainer = pl.Trainer(
#     max_epochs=config.experiment['epoch'],
    max_epochs=5,
    gpus=NGPU,
    weights_summary=config.experiment['weights_summary'],
    gradient_clip_val=config.experiment['gradient_clip'],
#     limit_train_batches=config.experiment['limit_train_batches'],  # coment in for training, running valiation every 30 batches
    limit_train_batches=1.0,  # coment in for training, running valiation every 30 batches
#     callbacks=[lr_logger, early_stop_callback],
    callbacks=[lr_logger],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=config.experiment['lr'][DATA], 
    hidden_size=config.model['hidden_size'], 
    attention_head_size=config.model['attention_head_size'],
    dropout=config.model['dropout'],
    hidden_continuous_size=config.model['hidden_continuous_size'],
    output_size=config.model['output_size'],  # 7 quantiles by default
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    log_interval=config.model['log_interval'],  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=config.model['reduce_on_plateau_patience'], # 4
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

# calcualte quantile loss on test set
best_tft.to(torch.device('cpu'))
actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
raw_predictions = best_tft.predict(test_dataloader, mode='raw')
raw_predictions = raw_predictions['prediction']
normalized_loss = normalized_quantile_loss(actuals, raw_predictions)
print(f'Normalized quantile loss - p10: {normalized_loss[0]}, p50: {normalized_loss[1]}, p90: {normalized_loss[2]}')