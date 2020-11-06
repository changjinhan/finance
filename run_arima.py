import pandas as pd
import numpy as np
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

data = preprocess('bitcoin')

ts = fill_missing_values(TimeSeries.from_dataframe(data, 'date', ['log_Close']), 'auto')
train, val = ts.split_after(pd.Timestamp('20201101'))

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



