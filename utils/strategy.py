from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import backtrader as bt
import locale
import pandas as pd
import numpy as np
import torch

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from preprocessing import preprocess
from utils.hparams import HParams


hparam_file = os.path.join(os.getcwd(), "hparams.yaml")
config = HParams.load(hparam_file)

class SmaCross(bt.Strategy):
    params = dict(
        pfast = 5, # period for the fast moving average
        pslow = 30 # period for the slow moving average
    )

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast) # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow) # slow moving average
        self.dataclose = self.datas[0].close # Keep a reference to the "close" line in the data[0] dataseries
        self.crossover = bt.ind.CrossOver(sma1, sma2) # crossover signal
        self.holding = 0 # number of shares held
        self.order = None # To keep track of pending orders

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                action = 'Buy'
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                action = 'Sell'
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
            # self.bar_executed = len(self)

            cash = self.broker.getcash()
            value = self.broker.getvalue()
            self.holding += order.size

            print('%s[%d] holding[%d] price[%d] cash[%.2f] value[%.2f]' % (action, abs(order.size), self.holding, self.dataclose[0], cash, value))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        if not self.position: # not in the market
            if self.crossover > 0: # if fast crosses slow to the upside
                available_stocks = int(self.broker.getcash()/self.dataclose[0]) # available buying size
                self.order = self.buy(size=available_stocks) # buying order
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

        elif self.crossover < 0: # in the market & cross to the downside
            self.order = self.close() # close the position
            self.log('SELL CREATE, %.2f' % self.dataclose[0])


class TFTpredict(bt.Strategy):
    params = dict(
        data = 'kospi', # period for the fast moving average
        symbol = '삼성전자' # period for the slow moving average
    )

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close # Keep a reference to the "close" line in the data[0] dataseries
        self.data = preprocess(self.p.data)
        self.symbol = self.p.symbol
        self.ckpt_path = config.transfer_path['9'][self.p.data]
        self.training = TimeSeriesDataSet(
                            self.data[self.data['Symbol'] == self.symbol],
                            time_idx=config.dataset_setting[self.p.data]['time_idx'],
                            target=config.dataset_setting[self.p.data]['target'],
                            group_ids=config.dataset_setting[self.p.data]['group_ids'],
                            min_encoder_length=config.experiment['max_encoder_length'][self.p.data] // 2,  # keep encoder length long (as it is in the validation set)
                            max_encoder_length=config.experiment['max_encoder_length'][self.p.data],
                            min_prediction_length=1,
                            max_prediction_length=config.experiment['max_prediction_length'],
                            static_categoricals=config.dataset_setting[self.p.data]['static_categoricals'],
                            static_reals=config.dataset_setting[self.p.data]['static_reals'],
                            time_varying_known_categoricals=config.dataset_setting[self.p.data]['time_varying_known_categoricals'],
                            variable_groups=config.dataset_setting[self.p.data]['variable_groups'],  # group of categorical variables can be treated as one variable
                            time_varying_known_reals=config.dataset_setting[self.p.data]['time_varying_known_reals'],
                            time_varying_unknown_categoricals=config.dataset_setting[self.p.data]['time_varying_unknown_categoricals'],
                            time_varying_unknown_reals=config.dataset_setting[self.p.data]['time_varying_unknown_reals'],
                            target_normalizer=GroupNormalizer(groups=config.dataset_setting[self.p.data]['group_ids']),  # normalize by group
                            allow_missings=True, # allow time_idx missing; Forward fill strategy
                            scalers={StandardScaler(): config.dataset_setting[self.p.data]['time_varying_unknown_reals']},
                            add_relative_time_idx=True,
                            add_target_scales=True,
                            add_encoder_length=True,
                        )
        self.model_total_length = config.experiment['max_encoder_length'][self.p.data] + config.experiment['max_prediction_length']
        self.model = TemporalFusionTransformer.load_from_checkpoint(self.ckpt_path)
        self.holding = 0 # number of shares held
        self.order = None # To keep track of pending orders

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                action = 'Buy'
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                action = 'Sell'
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
            # self.bar_executed = len(self)

            cash = self.broker.getcash()
            value = self.broker.getvalue()
            self.holding += order.size

            print('%s[%d] holding[%d] price[%d] cash[%.2f] value[%.2f]' % (action, abs(order.size), self.holding, self.dataclose[0], cash, value))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        test_data = self.data[self.data['Symbol'] == self.symbol]
        test_data = test_data[lambda x: pd.to_datetime(x.date) <= pd.to_datetime(self.datas[0].datetime.date(0))+timedelta(days=7)][-self.model_total_length:]
        testset = TimeSeriesDataSet.from_dataset(self.training, test_data, predict=True, stop_randomization=True)
        test_dataloader = testset.to_dataloader(train=False, batch_size=config.experiment['batch_size'], num_workers=0)
        predictions = self.model.predict(test_dataloader)

        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        if not self.position: # not in the market
            if (predictions > self.dataclose[0]).all(): # if all quantile 50 predictions are bigger than current close
                # print("predictions", predictions)
                # print("current", self.dataclose[0])
                available_stocks = int(self.broker.getcash()/self.dataclose[0]) # available buying size
                self.order = self.buy(size=available_stocks) # buying order
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

        elif (predictions < self.dataclose[0]).all(): # if all quantile 50 predictions are smaller than current close
            self.order = self.close() # close the position
            self.log('SELL CREATE, %.2f' % self.dataclose[0])