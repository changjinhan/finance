import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import copy

from preprocessing import preprocess_volatility
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer

from pytorch_forecasting.metrics import PoissonLoss, QuantileLoss, SMAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

warnings.filterwarnings("ignore")

# GPU allocation
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# data path
root_path = '/data3/finance/'
csv_path = os.path.join(root_path, 'oxfordmanrealizedvolatilityindices.csv')
output_file = os.path.join(root_path, 'volatility.csv')
asset_path = os.path.join(root_path, 'assets/cjhan/tft_logs')
optuna_path = os.path.join(root_path, 'assets/cjhan/tft_logs/optuna_test')

# preprocessing
data = preprocess_volatility(csv_path, output_file)

# Training setting
max_prediction_length = 5
max_encoder_length = 252
valid_boundary = '2016'
test_boundary = '2018'

training = TimeSeriesDataSet(
    data[lambda x: x.date < valid_boundary],
    time_idx='days_from_start',
    target="log_vol",
    group_ids=['Region', 'Symbol'],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["Region", 'Symbol'],
    static_reals=[],
    time_varying_known_categoricals=["day_of_week", "day_of_month", "week_of_year", "month"],
    variable_groups={},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["days_from_start"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["log_vol", "open_to_close"],
    target_normalizer=GroupNormalizer(groups=["Region", 'Symbol']),  # normalize by group
    allow_missings=True, # allow time_idx missing
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: (x.date >= valid_boundary) & (x.date < test_boundary)], predict=True, stop_randomization=True)
test = TimeSeriesDataSet.from_dataset(training, data[lambda x: x.date > test_boundary], predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 64  # set this between 32 to 128
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
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open(os.path.join(optuna_path, "test_study.pkl"), "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)

'''
{'gradient_clip_val': 0.3498626806469764, 'hidden_size': 78, 'dropout': 0.11547495781892213, 
'hidden_continuous_size': 26, 'attention_head_size': 3, 'learning_rate': 0.06901691924557539}                                                                                                        
'''






