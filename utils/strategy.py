from __future__ import (absolute_import, division, print_function, unicode_literals)

from datetime import datetime
import backtrader as bt
import locale

class SmaCross(bt.Strategy):
    params = dict(
        pfast = 5, # period for the fast moving average
        pslow = 30 # period for the slow moving average
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
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