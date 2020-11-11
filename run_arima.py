import pandas as pd
import numpy as np
from collections import defaultdict
from darts import TimeSeries
from darts.models import (
    FFT,
    AutoARIMA,
    ExponentialSmoothing,
    Prophet,
    Theta
)
from darts.metrics.metrics import mae, mape, rmse, smape 
from darts.utils.missing_values import fill_missing_values
from preprocessing import preprocess
import warnings
warnings.filterwarnings("ignore")

# data_list = ['btc_krw', 'btc_usd', 'crypto', 'vol']
data_list = ['vol']

for name in data_list:
    data = preprocess(name)
    if name == 'btc_krw':
        ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: x.date<='2020'], 'date', ['log_Close']), 'auto')
        train, val = ts.split_after(pd.Timestamp('20191226'))

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
        ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: x.date<='2018'], 'date', ['log_Close']), 'auto')
        train, val = ts.split_after(pd.Timestamp('20171226'))

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
                # Prophet(),
                # ExponentialSmoothing(),
                # Theta(),
                # FFT()
        ]

        for coin in coin_name:
            ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: (x.Symbol == coin) & (x.date<='2017')], 'date', ['log_Close']), 'auto')
            train, val = ts.split_after(pd.Timestamp('20161226'))
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

        models = [
                AutoARIMA(),
                # Prophet(),
                # ExponentialSmoothing(),
                # Theta(),
                # FFT()
        ]

        for sym in symbol_name:
            ts = fill_missing_values(TimeSeries.from_dataframe(data[lambda x: (x.Symbol == sym) & (x.date<='2018')], 'date', ['log_vol'], freq='B', fill_missing_dates=True), 'auto')
            train, val = ts.split_after(pd.Timestamp('20171221'))
            for model in models:
                model.fit(train)
                pred_val = model.predict(len(val))
                maes[str(model)].append(mae(pred_val, val))
                mapes[str(model)].append(mape(pred_val, val))
                rmses[str(model)].append(rmse(pred_val, val))
                smapes[str(model)].append(smape(pred_val, val))

        for model in models:
            print(str(model) + " MAE: " , np.mean(maes[str(model)]))
            print(str(model) + " MAPE: " , np.mean(mapes[str(model)]))
            print(str(model) + " RMSE: " , np.mean(rmses[str(model)]))
            print(str(model) + " SMAPE: " , np.mean(smapes[str(model)]))




