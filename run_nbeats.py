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
from preprocessing import preprocess
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

from pytorch_forecasting.metrics import PoissonLoss, QuantileLoss, SMAPE

warnings.filterwarnings("ignore")

# hyperparameter - using argparse and parameter module
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='experiment data', default='vol')
parser.add_argument('--idx', type=int, help='experiment number',  default=None)
parser.add_argument('--ws', type=str, help='machine number', default='9')
parser.add_argument('--gpu_index', '-g', type=int, default="0", help='GPU index')
parser.add_argument('--ngpu', type=int, default=0, help='0 = CPU.')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

# GPU allocation
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_index)

# device = torch.device("cuda:%d" % args.gpu_index if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
print('Current cuda device ', torch.cuda.current_device())
hparam_file = os.path.join(os.getcwd(), "hparams.yaml")

config = HParams.load(hparam_file)
asset_root = '/data3/finance/assets/cjhan/nbeats_logs/'

# seed
if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# preprocessing
data = preprocess(args.data)

# Training setting
max_prediction_length = 5
max_encoder_length = 90
valid_boundary = config.experiment['valid_boundary'][args.data]
test_boundary = config.experiment['test_boundary'][args.data]

training = TimeSeriesDataSet(
    data[lambda x: pd.to_datetime(x.date) < pd.to_datetime(valid_boundary)],
    time_idx=config.dataset_setting[args.data]['time_idx'],
    target=config.dataset_setting[args.data]['target'],
    group_ids=['Region', 'Symbol'],
    categorical_encoders={"Symbol": NaNLabelEncoder().fit(data.Symbol), "Region": NaNLabelEncoder().fit(data.Region)},
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length, # context length
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length, # prediction length
    time_varying_unknown_reals=['log_vol'],
    target_normalizer=GroupNormalizer(groups=config.dataset_setting[args.data]['group_ids']),  # normalize by group
    allow_missings=True, # allow time_idx missing
    scalers={StandardScaler(): ['log_vol']},
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

# calculate baseline absolute error
actuals = torch.cat([y for x, y in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
print('baseline SMAPE: ', SMAPE()(baseline_predictions, actuals))

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=10, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate

if not os.path.exists(asset_root):
    os.makedirs(asset_root)
logger = TensorBoardLogger(save_dir=asset_root, name=args.data, version=args.idx)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=100,
    gpus=args.ngpu,
    weights_summary=config.experiment['weights_summary'],
    gradient_clip_val=config.experiment['gradient_clip'],
    limit_train_batches=config.experiment['limit_train_batches'],  # coment in for training, running valiation every 30 batches
    callbacks=[lr_logger, early_stop_callback],
    # callbacks=[lr_logger],
    logger=logger,
)

net = NBeats.from_dataset(
    training,
    learning_rate=0.01,
    log_interval=config.model['log_interval'],
    log_val_interval=1,
    weight_decay=1e-2,
    widths=[32, 512],
    backcast_loss_ratio=1.0,
)

print(f"Number of parameters in network: {net.size()/1e3:.1f}k")

# fit network
trainer.fit(
    net,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"best model path: {best_model_path}")
best_nbeats = NBeats.load_from_checkpoint(best_model_path)

# calcualte SMAPE on validation set
actuals = torch.cat([y for x, y in iter(val_dataloader)])
predictions = best_nbeats.predict(val_dataloader)
print('validation SMAPE: ', SMAPE()(predictions, actuals))

# For testing, you should append test_step(), test_epoch_end() method in Basemodel class. (filepath: pytorch-forecasting > models > base_model.py)
# Test
trainer.test(
    best_nbeats,
    test_dataloaders=test_dataloader, 
    verbose=True,
)

# calcualte quantile loss on test set
best_nbeats.to(torch.device('cpu'))
actuals = torch.cat([y for x, y in iter(test_dataloader)])
raw_predictions = best_nbeats.predict(test_dataloader, mode='raw')
raw_predictions = raw_predictions['prediction']
print('test prediction: ', raw_predictions)
print('test actuals: ', actuals)
# normalized_loss = normalized_quantile_loss(actuals, raw_predictions)
# print(f'Normalized quantile loss - p10: {normalized_loss[0]}, p50: {normalized_loss[1]}, p90: {normalized_loss[2]}')

##### Visualizing Part #####
# image_root = os.path.join(logger.log_dir, 'images')
# if not os.path.exists(image_root):
#     os.makedirs(image_root)

# # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
# raw_predictions, x = best_tft.predict(test_dataloader, mode="raw", return_x=True)
# for idx in range(len(raw_predictions['groups'])): 
#     try:
#         fig = best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
#         fig.savefig(os.path.join(image_root, f'{args.data}_sample_{idx}.png'))
#     except:
#         continue

# # prediction plot sort by SMAPE 
# predictions = best_tft.predict(test_dataloader)
# mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
# # print('mean losses', mean_losses)
# indices = torch.flip(mean_losses.argsort(descending=True), (0,))  # sort losses
# # print('indices: ', indices)
# for idx in range(len(raw_predictions['groups'])): 
#     try:
#         fig2 = best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE())
#         fig2.savefig(os.path.join(image_root, f'{args.data}_best_SMAPE_{idx}.png'))
#     except:
#         continue

# predictions, x = best_tft.predict(test_dataloader, return_x=True)
# predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
# fig3_dict = best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
# for name in fig3_dict.keys():
#     fig3_dict[name].savefig(os.path.join(image_root, f'{args.data}_prediction_vs_actuals_{name}.png'))

# interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
# fig4_dict = best_tft.plot_interpretation(interpretation)
# for name in fig4_dict.keys():
#     fig4_dict[name].savefig(os.path.join(image_root, f'{args.data}_interpretation_{name}.png'))

