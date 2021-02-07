from datetime import datetime
from utils.strategy import SmaCross, TFTpredict
import backtrader as bt
import locale

stock_dict = {'삼성전자' : '005930.KS'}

# Samsung Electronics['005930.KS'] data
data = bt.feeds.YahooFinanceData(dataname=stock_dict['삼성전자'], fromdate=datetime(2020, 10, 21), todate=datetime(2021, 2, 6))

cerebro = bt.Cerebro() # create a "Cerebro" engine instance
cerebro.adddata(data)
cerebro.broker.setcash(1000000) # initial cash
cerebro.broker.setcommission(commission=0.00015) # commission 0.015%

cerebro.addstrategy(SmaCross) # add own strategy

start_value = cerebro.broker.getvalue()
cerebro.run() # backtesting start
final_value = cerebro.broker.getvalue()

print('* start value : %s won' % locale.format_string('%d', start_value, grouping=True))
print('* final value : %s won' % locale.format_string('%d', final_value, grouping=True))
print('* earning rate : %.2f %%' % ((final_value - start_value) / start_value * 100.0))

# cerebro.plot() # graph plotting
