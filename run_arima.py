import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from darts import TimeSeries
from darts.models import (
    FFT,
    AutoARIMA,
    ExponentialSmoothing,
    Prophet,
    Theta
)
from darts.metrics.metrics import mae, mape, rmse, smape, _get_values_or_raise
from darts.utils.missing_values import fill_missing_values
from preprocessing import preprocess
from utils.metrics import normalized_quantile_loss
import warnings
warnings.filterwarnings("ignore")

data_list = ['kospi200']

for name in data_list:
    print('-'*30, name)
    data = preprocess(name)
    if name == 'btc_krw':
        ts = fill_missing_values(TimeSeries.from_dataframe(data, 'date', ['Close']), 'auto')
        train, val = ts.split_after(pd.Timestamp(list(data['date'])[-6]))

        models = [
            AutoARIMA(),
            Prophet(),
            ExponentialSmoothing(),
            Theta(),
            FFT()
        ]

        for model in models:
            model.fit(train)
            pred_val = model.predict(len(val))
            print(str(model) + " MAE: " + str(mae(pred_val,val)))
            print(str(model) + " MAPE: " + str(mape(pred_val, val)))
            print(str(model) + " RMSE: " + str(rmse(pred_val, val)))
            print(str(model) + " SMAPE: " + str(smape(pred_val, val)))

    elif name == 'btc_usd':
        ts = fill_missing_values(TimeSeries.from_dataframe(data, 'date', ['Close']), 'auto')
        train, val = ts.split_after(pd.Timestamp(pd.Timestamp(list(data['date'])[-6])))

        models = [
            AutoARIMA(),
            Prophet(),
            ExponentialSmoothing(),
            Theta(),
            FFT()
        ]

        for model in models:
            model.fit(train)
            pred_val = model.predict(len(val))
            print(str(model) + " MAE: " + str(mae(pred_val,val)))
            print(str(model) + " MAPE: " + str(mape(pred_val, val)))
            print(str(model) + " RMSE: " + str(rmse(pred_val, val)))
            print(str(model) + " SMAPE: " + str(smape(pred_val, val)))
    
    elif name == 'crypto':
        coin_name = data['Symbol'].unique()
        maes = defaultdict(list)
        mapes = defaultdict(list)
        rmses = defaultdict(list)
        smapes = defaultdict(list)

        models = [
                AutoARIMA(),
                Prophet(),
                ExponentialSmoothing(),
                Theta(),
                FFT()
        ]

        for coin in coin_name:
            coin_data = data[lambda x: x.Symbol == coin]
            ts = fill_missing_values(TimeSeries.from_dataframe(coin_data, 'date', ['Close']), 'auto')
            train, val = ts.split_after(pd.Timestamp(list(coin_data['date'])[-6]))
            for model in models:
                model.fit(train)
                pred_val = model.predict(len(val))
                maes[str(model)].append(mae(pred_val,val))
                mapes[str(model)].append(mape(pred_val, val))
                rmses[str(model)].append(rmse(pred_val, val))
                smapes[str(model)].append(smape(pred_val, val))

        for model in models:
            print(str(model) + " MAE: " , np.mean(maes[str(model)]))
            print(str(model) + " MAPE: " , np.mean(mapes[str(model)]))
            print(str(model) + " RMSE: " , np.mean(rmses[str(model)]))
            print(str(model) + " SMAPE: " , np.mean(smapes[str(model)]))

    elif name == 'crypto_hourly':
        coin_name = data['Symbol'].unique()
        maes = defaultdict(list)
        mapes = defaultdict(list)
        rmses = defaultdict(list)
        smapes = defaultdict(list)

        models = [
                AutoARIMA(),
                Prophet(),
                ExponentialSmoothing(),
                Theta(),
                FFT()
        ]

        for coin in coin_name:
            coin_data = data[lambda x: x.Symbol == coin]
            ts = fill_missing_values(TimeSeries.from_dataframe(coin_data, 'date', ['Close']), 'auto')
            train, val = ts.split_after(pd.Timestamp(list(coin_data['date'])[-6]))
            for model in models:
                model.fit(train)
                pred_val = model.predict(len(val))
                maes[str(model)].append(mae(pred_val,val))
                mapes[str(model)].append(mape(pred_val, val))
                rmses[str(model)].append(rmse(pred_val, val))
                smapes[str(model)].append(smape(pred_val, val))

        for model in models:
            print(str(model) + " MAE: " , np.mean(maes[str(model)]))
            print(str(model) + " MAPE: " , np.mean(mapes[str(model)]))
            print(str(model) + " RMSE: " , np.mean(rmses[str(model)]))
            print(str(model) + " SMAPE: " , np.mean(smapes[str(model)]))

    elif name == 'vol':
        symbol_name = data['Symbol'].unique()
        maes = defaultdict(list)
        mapes = defaultdict(list)
        rmses = defaultdict(list)
        smapes = defaultdict(list)
        quantile_losses = defaultdict(list)

        models = [
                AutoARIMA(alpha=0.8), # for quantile 0.1, 0.9 
                # Prophet(),
                # ExponentialSmoothing(),
                # Theta(),
                # FFT()
        ]
        quantiles = [0.1, 0.5, 0.9]
        for sym in symbol_name:
            ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: (x.Symbol == sym) & (pd.to_datetime(x.date)<=pd.to_datetime('20190629'))], 'date', ['log_vol'], freq='B', fill_missing_dates=True), 'auto')
            train, val = ts.split_after(pd.Timestamp('20190621'))
            train = train[-252:]
            for model in models:
                model.fit(train)
                pred_val, conf_int = model.predict(len(val), return_conf_int=True)
                # print(conf_int)
                maes[str(model)].append(mae(pred_val, val))
                mapes[str(model)].append(mape(pred_val, val))
                rmses[str(model)].append(rmse(pred_val, val))
                smapes[str(model)].append(smape(pred_val, val))

                pred_val, val = _get_values_or_raise(pred_val, val, True)
                pred_val = np.column_stack([conf_int[...,0], pred_val, conf_int[...,1]]) # quantile regression for q=[0.1, 0.5, 0.9]

                # calculate quantile loss
                q_losses = []
                normalizer = np.sum(abs(val))
                for i, q in enumerate(quantiles):
                    # print(pred_val[..., i])
                    errors = val - pred_val[..., i]
                    q_losses.append((np.maximum((q - 1) * errors, q * errors)))
                normalized_q_loss = 2 * np.sum(q_losses, axis=1) / normalizer
                quantile_losses[str(model)].append(normalized_q_loss)

        for model in models:
            print(str(model) + " MAE: " , np.mean(maes[str(model)]))
            print(str(model) + " MAPE: " , np.mean(mapes[str(model)]))
            print(str(model) + " RMSE: " , np.mean(rmses[str(model)]))
            print(str(model) + " SMAPE: " , np.mean(smapes[str(model)]))
            print(str(model) + " Quantile Loss: ", np.mean(quantile_losses[str(model)], axis=0))

    elif name == 'sp500':
        symbol_name = data['Symbol'].unique()
        maes = defaultdict(list)
        mapes = defaultdict(list)
        rmses = defaultdict(list)
        smapes = defaultdict(list)
        quantile_losses = defaultdict(list)

        models = [
                AutoARIMA(alpha=0.8), # for quantile 0.1, 0.9 
                # Prophet(),
                # ExponentialSmoothing(),
                # Theta(),
                # FFT()
        ]
        quantiles = [0.1, 0.5, 0.9]
        for sym in symbol_name:
            ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: (x.Symbol == sym)], 'date', ['Close'], freq='B', fill_missing_dates=True), 'auto')
            train, val = ts.split_after(pd.Timestamp('20201211'))
            # print(len(val))
            train = train[-252:]
            for model in models:
                model.fit(train)
                pred_val, conf_int = model.predict(len(val), return_conf_int=True)
                # print(conf_int)
                maes[str(model)].append(mae(pred_val, val))
                mapes[str(model)].append(mape(pred_val, val))
                rmses[str(model)].append(rmse(pred_val, val))
                smapes[str(model)].append(smape(pred_val, val))

                pred_val, val = _get_values_or_raise(pred_val, val, True)
                pred_val = np.column_stack([conf_int[...,0], pred_val, conf_int[...,1]]) # quantile regression for q=[0.1, 0.5, 0.9]

                # calculate quantile loss
                q_losses = []
                normalizer = np.sum(abs(val))
                for i, q in enumerate(quantiles):
                    # print(pred_val[..., i])
                    errors = val - pred_val[..., i]
                    q_losses.append((np.maximum((q - 1) * errors, q * errors)))
                normalized_q_loss = 2 * np.sum(q_losses, axis=1) / normalizer
                quantile_losses[str(model)].append(normalized_q_loss)

        for model in models:
            print(str(model) + " MAE: " , np.mean(maes[str(model)]))
            print(str(model) + " MAPE: " , np.mean(mapes[str(model)]))
            print(str(model) + " RMSE: " , np.mean(rmses[str(model)]))
            print(str(model) + " SMAPE: " , np.mean(smapes[str(model)]))
            print(str(model) + " Quantile Loss: ", np.mean(quantile_losses[str(model)], axis=0))

    elif name == 'kospi':
        symbol_name = data['Symbol'].unique()
        maes = defaultdict(list)
        mapes = defaultdict(list)
        rmses = defaultdict(list)
        smapes = defaultdict(list)
        quantile_losses = defaultdict(list)

        models = [
                AutoARIMA(alpha=0.8), # for quantile 0.1, 0.9 
                # Prophet(),
                # ExponentialSmoothing(),
                # Theta(),
                # FFT()
        ]
        quantiles = [0.1, 0.5, 0.9]
        for sym in symbol_name:
            try:
                ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime('2010')) & (x.Symbol == sym)], 'date', ['Close'], freq='B', fill_missing_dates=True), 'auto')
                train, val = ts.split_after(pd.Timestamp('20201211'))
                # print(len(val))
                train = train[-252:]
                for model in models:
                    model.fit(train)
                    pred_val, conf_int = model.predict(len(val), return_conf_int=True)
            except ValueError as e:
                ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: (x.Symbol == sym)], 'date', ['Close'], freq='B', fill_missing_dates=True), 0.)
                train, val = ts.split_after(pd.Timestamp('20201211'))
                train = train[-252:]
                for model in models:
                    model.fit(train)
                    pred_val, conf_int = model.predict(len(val), return_conf_int=True)

            maes[str(model)].append(mae(pred_val, val))
            mapes[str(model)].append(mape(pred_val, val))
            rmses[str(model)].append(rmse(pred_val, val))
            smapes[str(model)].append(smape(pred_val, val))

            pred_val, val = _get_values_or_raise(pred_val, val, True)
            pred_val = np.column_stack([conf_int[...,0], pred_val, conf_int[...,1]]) # quantile regression for q=[0.1, 0.5, 0.9]

            # calculate quantile loss
            q_losses = []
            normalizer = np.sum(abs(val))
            for i, q in enumerate(quantiles):
                # print(pred_val[..., i])
                errors = val - pred_val[..., i]
                q_losses.append((np.maximum((q - 1) * errors, q * errors)))
            normalized_q_loss = 2 * np.sum(q_losses, axis=1) / normalizer
            quantile_losses[str(model)].append(normalized_q_loss)

        for model in models:
            print(str(model) + " MAE: " , np.mean(maes[str(model)]))
            print(str(model) + " MAPE: " , np.mean(mapes[str(model)]))
            print(str(model) + " RMSE: " , np.mean(rmses[str(model)]))
            print(str(model) + " SMAPE: " , np.mean(smapes[str(model)]))
            print(str(model) + " Quantile Loss: ", np.mean(quantile_losses[str(model)], axis=0))

    elif name == 'kospi200':
        symbol_name = data['Symbol'].unique()
        maes = defaultdict(list)
        mapes = defaultdict(list)
        rmses = defaultdict(list)
        smapes = defaultdict(list)
        quantile_losses = defaultdict(list)

        models = [
                AutoARIMA(alpha=0.8), # for quantile 0.1, 0.9 
                # Prophet(),
                # ExponentialSmoothing(),
                # Theta(),
                # FFT()
        ]
        quantiles = [0.1, 0.5, 0.9]
        normalizer, normalized_q_loss = 0, 0
        for sym in symbol_name:
            try:
                ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: (pd.to_datetime(x.date) >= pd.to_datetime('2010')) & (x.Symbol == sym)], 'date', ['Close'], freq='B', fill_missing_dates=True), 'auto')
                train, val = ts.split_after(pd.Timestamp('20210219'))
                # print(len(val))
                train = train[-252:]
                for model in models:
                    model.fit(train)
                    pred_val, conf_int = model.predict(len(val), return_conf_int=True)
            except ValueError as e:
                ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: (x.Symbol == sym)], 'date', ['Close'], freq='B', fill_missing_dates=True), 0.)
                train, val = ts.split_after(pd.Timestamp('20210219'))
                train = train[-252:]
                for model in models:
                    model.fit(train)
                    pred_val, conf_int = model.predict(len(val), return_conf_int=True)

            maes[str(model)].append(mae(pred_val, val))
            mapes[str(model)].append(mape(pred_val, val))
            rmses[str(model)].append(rmse(pred_val, val))
            smapes[str(model)].append(smape(pred_val, val))

            pred_val, val = _get_values_or_raise(pred_val, val, True)
            pred_val = np.column_stack([conf_int[...,0], pred_val, conf_int[...,1]]) # quantile regression for q=[0.1, 0.5, 0.9]
            
            # calculate quantile loss
            q_losses = []
            normalizer += np.sum(abs(val))
            for i, q in enumerate(quantiles):
                # print(pred_val[..., i])
                errors = val - pred_val[..., i]
                q_losses.append((np.maximum((q - 1) * errors, q * errors)))
            normalized_q_loss += 2 * np.sum(q_losses, axis=1)
            # quantile_losses[str(model)].append(normalized_q_loss)
        normalized_q_loss = normalized_q_loss / normalizer

        for model in models:
            print(str(model) + " MAE: " , np.mean(maes[str(model)]))
            print(str(model) + " MAPE: " , np.mean(mapes[str(model)]))
            print(str(model) + " RMSE: " , np.mean(rmses[str(model)]))
            print(str(model) + " SMAPE: " , np.mean(smapes[str(model)]))
            # print(str(model) + " Quantile Loss: ", np.mean(quantile_losses[str(model)], axis=0))
            print(str(model) + " Quantile Loss: ", normalized_q_loss)


