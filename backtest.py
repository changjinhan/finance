import os
import argparse
import backtrader as bt
import locale
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, date
from utils.hparams import HParams
from utils.strategy import SmaCross, TFTpredict


hparam_file = os.path.join(os.getcwd(), "hparams.yaml")
config = HParams.load(hparam_file)

def backtest(dataset, model, loss, stock_name, symbol, from_date, to_date, strategy):
    # data load
    data = bt.feeds.YahooFinanceData(dataname=symbol, fromdate=date.fromisoformat(from_date), todate=date.fromisoformat(to_date))

    cerebro = bt.Cerebro() # create a "Cerebro" engine instance
    cerebro.adddata(data)
    cerebro.broker.setcash(10000000) # initial cash
    cerebro.broker.setcommission(commission=0.00015) # commission 0.015%

    # strategy setting
    if strategy == 'SMA':
        my_strategy = SmaCross
    else:
        my_strategy = TFTpredict
        my_strategy.params.data = dataset
        my_strategy.params.model = model
        my_strategy.params.loss = loss
        my_strategy.params.symbol = stock_name

    cerebro.addstrategy(my_strategy) # add own strategy

    start_value = cerebro.broker.getvalue()
    cerebro.run() # backtesting start
    final_value = cerebro.broker.getvalue()
    earning_rate = (final_value - start_value) / start_value * 100.0

    print('* start value : %s won' % locale.format_string('%d', start_value, grouping=True))
    print('* final value : %s won' % locale.format_string('%d', final_value, grouping=True))
    print('* earning rate : %.2f %%' % earning_rate)

    # cerebro.plot() # graph plotting
    return earning_rate

# hyperparameter - using argparse and parameter module
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='experiment data', default='kospi200+TI')
parser.add_argument('--model', type=str, help='forecasting model(TFT, entmax15, sparsemax)', default='TFT')
parser.add_argument('--loss', type=str, help='loss function', default='quantile')
parser.add_argument('--strategy', type=str, help='trading strategy(SMA, TFT)', default='TFT')
args = parser.parse_args()

stock_list = config.backtest['9']['data'][args.data]['model'][args.model]['loss'][args.loss]['stocks']

# kospi stock listing to find the symbol of stock
kospi = fdr.StockListing('KOSPI')
returns = []

for i, stock_name in enumerate(stock_list):
    symbol = kospi['Symbol'][kospi['Name'] == stock_name].values[0] + '.KS'
    from_date = config.backtest['from_date']
    to_date = config.backtest['to_date']
    print(f'{stock_name} | {symbol}')
    earning_rate = backtest(args.data, args.model, args.loss, stock_name, symbol, from_date, to_date, args.strategy)
    returns.append(earning_rate)

print(returns)
print(np.mean(returns))




