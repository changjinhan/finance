import os
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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# data path
root_path = '/data3/finance/'
csv_path = os.path.join(root_path, 'oxfordmanrealizedvolatilityindices.csv')
output_file = os.path.join(root_path, 'volatility.csv')
asset_path = os.path.join(root_path, 'assets/cjhan/tft_logs')

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
test = TimeSeriesDataSet.from_dataset(training, data[lambda x: (x.date > test_boundary) & (x.date < '2019.06.29')], predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 64  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, y in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
print('baseline MAE: ', (actuals - baseline_predictions).abs().mean().item())

# configure network and trainer
# early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=10, verbose=True, mode="min")
# lr_logger = LearningRateMonitor()  # log the learning rate

if not os.path.exists(asset_path):
    os.makedirs(asset_path)
logger = TensorBoardLogger(asset_path)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=100,
    gpus=1,
    weights_summary="top",
    gradient_clip_val=0.3498626806469764, # 0.1
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    # callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.06901691924557539, # 0.01
    hidden_size=78, # 160
    attention_head_size=3, # 1
    dropout=0.11547495781892213, # 0.3
    hidden_continuous_size=26, # 8
    output_size=3,  # 7 quantiles by default
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    # log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    # reduce_on_plateau_patience=4,
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

