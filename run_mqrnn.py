import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.MQRNN import Encoder
from utils.MQRNN import Decoder
from utils.MQRNN.MQRNN import MQRNN 
from utils.MQRNN.data import MQRNN_dataset,read_df
from utils.MQRNN.data import MQRNN_dataset
from utils.metrics import normalized_quantile_loss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('/data3/finance/kospi200.csv')
symbol_list = df['Symbol'].unique()

config = {
    'horizon_size': 5,
    'hidden_size': 50,
    'columns': [1],
    'quantiles': [0.1, 0.5, 0.9], 
    'dropout': 0.3,
    'layer_size':4,
    'by_direction':False,
    'lr': 1e-2,
    'batch_size': 128,
    'num_epochs': 50,
    'context_size': 10,
}

normalizer, normalized_q_loss = 0, 0
for i, symbol in enumerate(symbol_list):
    print('-'*50)
    print(symbol, f'({i+1}/{len(symbol_list)})')
    config['symbol'] = symbol

    # data load
    train_target_df, test_target_df, train_covariate_df, test_covariate_df = read_df(config)
    config['covariate_size'] = train_covariate_df.shape[1]

    
    # MQRNN model
    net = MQRNN(config['horizon_size'],
                config['hidden_size'],
                config['quantiles'],
                config['columns'],
                config['dropout'],
                config['layer_size'],
                config['by_direction'],
                config['lr'],
                config['batch_size'],
                config['num_epochs'],
                config['context_size'],
                config['covariate_size'],
                device)
    
    # train dataset
    train_dataset = MQRNN_dataset(train_target_df, train_covariate_df, config['horizon_size'], len(config['quantiles']))

    # training
    net.train(train_dataset)

    # predict
    predict_result = net.predict(train_target_df,train_covariate_df,test_covariate_df,'close')
    pred_val = np.row_stack([predict_result[0.1], predict_result[0.5], predict_result[0.9]])
    pred_val = np.transpose(pred_val)
    val = test_target_df['close'].to_numpy()

    # calculate quantile loss
    q_losses = []
    normalizer += np.sum(abs(val))
    for i, q in enumerate(config['quantiles']):
        errors = val - pred_val[..., i]
        q_losses.append((np.maximum((q - 1) * errors, q * errors)))
    normalized_q_loss += 2 * np.sum(q_losses, axis=1)
normalized_q_loss = normalized_q_loss / normalizer

print("Normalized Quantile Loss: ", normalized_q_loss)