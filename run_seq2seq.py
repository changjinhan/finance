import os
import warnings
import numpy as np
import pandas as pd
import random
import argparse
import mxnet as mx
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
from mxnet import nd, gpu, gluon, autograd
from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.seq2seq import Seq2SeqEstimator,RNN2QRForecaster, MQCNNEstimator
from gluonts.mx.trainer._base import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.block.encoder import MLPEncoder, RNNEncoder
from gluonts.mx.block.scaler import NOPScaler
from gluonts.mx.trainer.model_averaging import SelectNBestMean
from gluonts.evaluation import Evaluator
from utils.hparams import HParams
from preprocessing import preprocess

warnings.filterwarnings("ignore")
mx.nd.waitall() # error handling

# hyperparameter - using argparse and parameter module
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='experiment data', default='kospi200+TI')
parser.add_argument('--ws', type=str, help='machine number', default='9')
parser.add_argument('--gpu_index', '-g', type=int, help='GPU index', default=0)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# seed
if args.seed > 0:
    np.random.seed(args.seed)
    random.seed(args.seed)

# setting for print out korean in figures 
plt.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

hparam_file = os.path.join(os.getcwd(), "hparams.yaml")
config = HParams.load(hparam_file)

# preprocessing
dataset = preprocess(args.data)

# custom dataset
train_boundary = config.experiment['train_boundary'][args.data]
prediction_length = config.experiment['max_prediction_length']
freq="B"
start = pd.Timestamp("2010-01-04", freq=freq)

normalized_quantile_loss = []
symbol_name = dataset["Symbol"].unique()[100:]

for i, sym in enumerate(symbol_name):
    print('-'*50)
    print(sym, f'({i+1}/{len(symbol_name)})')
    data = dataset[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime(train_boundary)) & (x.Symbol == sym)]
    
    train_ds = ListDataset([{
        'target': data[:-prediction_length]['Close'], 
        'start': start,
        # 'feat_static_cat': np.zeros(len(data[:-prediction_length])),
        # 'feat_static_real':,
        # 'feat_dynamic_cat':,
        # 'feat_dynamic_real':,
        }], freq=freq)

    test_ds = ListDataset([{
        'target': data['Close'], 
        'start': start,
        # 'feat_static_cat': np.zeros(len(data)),
        }], freq=freq)

    # estimator = Seq2SeqEstimator(
    #     prediction_length = prediction_length,
    #     embedding_dimension = 50,
    #     context_length = config.experiment['max_encoder_length'][args.data],
    #     freq = freq,
    #     cardinality= [50],
    #     encoder = RNNEncoder(
    #                 mode='lstm', 
    #                 hidden_size=50,
    #                 num_layers=1,
    #                 bidirectional=False,
    #                 use_static_feat=False,
    #                 use_dynamic_feat=False,
    #             ),
    #     # encoder = MLPEncoder(
    #     #                 layer_sizes=[16,16,16],
    #     #             ),
    #     decoder_mlp_layer= [10],
    #     decoder_mlp_static_dim= 50,
    #     scaler= NOPScaler(),
    #     trainer=Trainer(
    #                 batch_size=config.experiment['batch_size'],
    #                 epochs=config.experiment['epoch'],
    #                 learning_rate=config.experiment['lr'][args.data],
    #                 ctx=gpu(args.gpu_index),
    #             )
    # )

    # estimator = RNN2QRForecaster(
    #     freq = freq, 
    #     prediction_length = prediction_length, 
    #     cardinality = [1], 
    #     embedding_dimension = 50, 
    #     encoder_rnn_layer = 1, 
    #     encoder_rnn_num_hidden = 50, 
    #     decoder_mlp_layer = [10], 
    #     decoder_mlp_static_dim = 50, 
    #     encoder_rnn_model = 'lstm', 
    #     encoder_rnn_bidirectional = False, 
    #     scaler = NOPScaler(), 
    #     context_length = config.experiment['max_encoder_length'][args.data], 
    #     trainer = Trainer(
    #         avg_strategy=SelectNBestMean(maximize=False, metric="score", num_models=1), 
    #         batch_size=config.experiment['batch_size'], 
    #         clip_gradient=10.0, 
    #         ctx=gpu(args.gpu_index), 
    #         epochs=config.experiment['epoch'], 
    #         hybridize=True, 
    #         init="xavier", 
    #         learning_rate=config.experiment['lr'][args.data], 
    #         learning_rate_decay_factor=0.5, 
    #         minimum_learning_rate=5e-05, 
    #         patience=10,
    #         post_initialize_cb=None, 
    #         weight_decay=1e-08
    #     )
    # )

    # estimator = SimpleFeedForwardEstimator(
    #     num_hidden_dimensions=[64,64,64],
    #     prediction_length=prediction_length,
    #     context_length=config.experiment['max_encoder_length'][args.data],
    #     freq=freq,
    #     trainer=Trainer(ctx=gpu(args.gpu_index), 
    #                     epochs=config.experiment['epoch'], 
    #                     learning_rate=config.experiment['lr'][args.data], 
    #                     num_batches_per_epoch=1000
    #     )
    # )

    estimator = MQCNNEstimator(
        prediction_length = prediction_length,
        context_length = config.experiment['max_encoder_length'][args.data],
        freq = freq,
        quantiles=[0.1,0.5,0.9],
        trainer=Trainer(
                    batch_size=config.experiment['batch_size'],
                    epochs=config.experiment['epoch'],
                    # epochs=1,
                    learning_rate=config.experiment['lr'][args.data],
                    ctx=gpu(args.gpu_index),
                )
    )

    predictor = estimator.train(train_ds)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    # first entry of test dataset
    dataset_test_entry = next(iter(test_ds))
    actuals = dataset_test_entry['target'][-prediction_length:]
    # print(f"actual values of the future window:\n {actuals}")

    # first entry of the forecast list
    forecast_entry = forecasts[0]
    # print(f"0.1-quantile of the future window:\n {forecast_entry.quantile(0.1)}")
    # print(f"0.5-quantile of the future window:\n {forecast_entry.quantile(0.5)}")
    # print(f"0.9-quantile of the future window:\n {forecast_entry.quantile(0.9)}")

    # compute metrics
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    # print(json.dumps(agg_metrics, indent=4))

    # normalized quantile loss
    # print(2*agg_metrics['QuantileLoss[0.1]']/sum(actuals))
    # print(2*agg_metrics['QuantileLoss[0.5]']/sum(actuals))
    # print(2*agg_metrics['QuantileLoss[0.9]']/sum(actuals))

    normalized_quantile_loss.append(np.array([2*agg_metrics['QuantileLoss[0.1]']/sum(actuals),
                                            2*agg_metrics['QuantileLoss[0.5]']/sum(actuals),
                                            2*agg_metrics['QuantileLoss[0.9]']/sum(actuals)]))

print(np.mean(normalized_quantile_loss, axis=0))
