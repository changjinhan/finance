import os
import warnings
import numpy as np
import pandas as pd
import copy
import FinanceDataReader as fdr

from ta.utils import dropna
from ta.trend import MACD, PSARIndicator
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator, ROCIndicator
from ta.volume import OnBalanceVolumeIndicator, ForceIndexIndicator

warnings.filterwarnings("ignore")


def preprocess(data_name, symbol=None):
    output_file = os.path.join('/data3/finance/', data_name + '.csv')
    if data_name == 'vol':
        ''' volatility '''
        if not os.path.exists(output_file):
            csv_path = '/data3/finance/oxfordmanrealizedvolatilityindices.csv'
            data = pd.read_csv(csv_path, encoding='utf-8')
            data.rename(columns={'Unnamed: 0':'Datetime'}, inplace=True)
            # Adds additional date/day fields
            dt = [str(s).split('+')[0] for s in data['Datetime']]  # ignore timezones, we don't need them
            dates = pd.to_datetime(dt)
            data['date'] = dates
            data['days_from_start'] = (dates - pd.datetime(2000, 1, 3)).days
            data['day_of_week'] = dates.dayofweek.astype(str).astype('category')
            data['day_of_month'] = dates.day.astype(str).astype('category')
            data['week_of_year'] = dates.weekofyear.astype(str).astype('category')
            data['month'] = dates.month.astype(str).astype('category')
            data['year'] = dates.year.astype(str).astype('category')

            # Processes log volatility
            vol = data['rv5_ss'].copy()
            vol.loc[vol == 0.] = np.nan
            data['log_vol'] = np.log(vol)

            # Adds static information
            symbol_region_mapping = {
            '.AEX': 'EMEA',
            '.AORD': 'APAC',
            '.BFX': 'EMEA',
            '.BSESN': 'APAC',
            '.BVLG': 'EMEA',
            '.BVSP': 'AMER',
            '.DJI': 'AMER',
            '.FCHI': 'EMEA',
            '.FTMIB': 'EMEA',
            '.FTSE': 'EMEA',
            '.GDAXI': 'EMEA',
            '.GSPTSE': 'AMER',
            '.HSI': 'APAC',
            '.IBEX': 'EMEA',
            '.IXIC': 'AMER',
            '.KS11': 'APAC',
            '.KSE': 'APAC',
            '.MXX': 'AMER',
            '.N225': 'APAC',
            '.NSEI': 'APAC',
            '.OMXC20': 'EMEA',
            '.OMXHPI': 'EMEA',
            '.OMXSPI': 'EMEA',
            '.OSEAX': 'EMEA',
            '.RUT': 'EMEA',
            '.SMSI': 'EMEA',
            '.SPX': 'AMER',
            '.SSEC': 'APAC',
            '.SSMI': 'EMEA',
            '.STI': 'APAC',
            '.STOXX50E': 'EMEA'
            }

            data['Region'] = data['Symbol'].apply(lambda k: symbol_region_mapping[k])

            # Performs final processing
            output_df_list = []
            for grp in data.groupby('Symbol'):
                sliced = grp[1].copy()
                sliced.sort_values('days_from_start', inplace=True)
                # Impute log volatility values
                sliced['log_vol'].fillna(method='ffill', inplace=True)
                sliced.dropna()
                sliced['time_idx'] = np.arange(len(sliced))
                output_df_list.append(sliced)

            data = pd.concat(output_df_list, axis=0)

            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')
        else:
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')
    
    elif data_name == 'stock_idx':
        ''' stock index '''
        if not os.path.exists(output_file):
            csv_path = '/data3/finance/oxfordmanrealizedvolatilityindices.csv'
            data = pd.read_csv(csv_path, encoding='utf-8')
            data.rename(columns={'Unnamed: 0':'Datetime'}, inplace=True)
            # Adds additional date/day fields
            dt = [str(s).split('+')[0] for s in data['Datetime']]  # ignore timezones, we don't need them
            dates = pd.to_datetime(dt)
            data['date'] = dates
            data['days_from_start'] = (dates - pd.datetime(2000, 1, 3)).days
            data['day_of_week'] = dates.dayofweek.astype(str).astype('category')
            data['day_of_month'] = dates.day.astype(str).astype('category')
            data['week_of_year'] = dates.weekofyear.astype(str).astype('category')
            data['month'] = dates.month.astype(str).astype('category')
            data['year'] = dates.year.astype(str).astype('category')

            # Processes log close & log volatility
            close = data['close_price'].copy()
            close.loc[close == 0.] = np.nan
            data['log_close'] = np.log(close)
            vol = data['rv5_ss'].copy()
            vol.loc[vol == 0.] = np.nan
            data['log_vol'] = np.log(vol)

            # Adds static information
            symbol_region_mapping = {
            '.AEX': 'EMEA',
            '.AORD': 'APAC',
            '.BFX': 'EMEA',
            '.BSESN': 'APAC',
            '.BVLG': 'EMEA',
            '.BVSP': 'AMER',
            '.DJI': 'AMER',
            '.FCHI': 'EMEA',
            '.FTMIB': 'EMEA',
            '.FTSE': 'EMEA',
            '.GDAXI': 'EMEA',
            '.GSPTSE': 'AMER',
            '.HSI': 'APAC',
            '.IBEX': 'EMEA',
            '.IXIC': 'AMER',
            '.KS11': 'APAC',
            '.KSE': 'APAC',
            '.MXX': 'AMER',
            '.N225': 'APAC',
            '.NSEI': 'APAC',
            '.OMXC20': 'EMEA',
            '.OMXHPI': 'EMEA',
            '.OMXSPI': 'EMEA',
            '.OSEAX': 'EMEA',
            '.RUT': 'EMEA',
            '.SMSI': 'EMEA',
            '.SPX': 'AMER',
            '.SSEC': 'APAC',
            '.SSMI': 'EMEA',
            '.STI': 'APAC',
            '.STOXX50E': 'EMEA'
            }

            data['Region'] = data['Symbol'].apply(lambda k: symbol_region_mapping[k])

            # Performs final processing
            output_df_list = []
            for grp in data.groupby('Symbol'):
                sliced = grp[1].copy()
                sliced.sort_values('days_from_start', inplace=True)
                # Impute log close values
                sliced['log_close'].fillna(method='ffill', inplace=True)
                sliced['log_vol'].fillna(method='ffill', inplace=True)
                sliced.dropna()
                sliced['time_idx'] = np.arange(len(sliced))
                output_df_list.append(sliced)

            data = pd.concat(output_df_list, axis=0)

            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')

            if symbol is not None:
                data = data[data['Symbol'] == symbol] # test for each stock index symbol 
        else:
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')
            
            if symbol is not None:
                data = data[data['Symbol'] == symbol] # test for each stock index symbol

    elif data_name == 'btc_krw':
        if not os.path.exists(output_file):
            data = fdr.DataReader('BTC/KRW')
            data.reset_index(inplace=True)
            data.rename(columns={'Date':'date'}, inplace=True)
            # data preprocessing
            dates = pd.to_datetime(data['date'].to_list())
            data['days_from_start'] = (dates - pd.datetime(2017, 5, 23)).days
            data['day_of_week'] = dates.dayofweek.astype(str).astype('category')
            data['day_of_month'] = dates.day.astype(str).astype('category')
            data['week_of_year'] = dates.weekofyear.astype(str).astype('category')
            data['month'] = dates.month.astype(str).astype('category')
            data['year'] = dates.year.astype(str).astype('category')

            # Processes log Close
            close = data['Close'].copy()
            data['log_Close'] = np.log(close)
            
            data['Symbol'] = 'BTC/KRW'
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')
        else:
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')
    
    elif data_name == 'btc_usd':
        if not os.path.exists(output_file):
            data = fdr.DataReader('BTC/USD')
            data.reset_index(inplace=True)
            data.rename(columns={'Date':'date'}, inplace=True)
            # data preprocessing
            dates = pd.to_datetime(data['date'].to_list())
            data['days_from_start'] = (dates - pd.datetime(2010, 7, 18)).days
            data['day_of_week'] = dates.dayofweek.astype(str).astype('category')
            data['day_of_month'] = dates.day.astype(str).astype('category')
            data['week_of_year'] = dates.weekofyear.astype(str).astype('category')
            data['month'] = dates.month.astype(str).astype('category')
            data['year'] = dates.year.astype(str).astype('category')

            # Processes log Close
            close = data['Close'].copy()
            data['log_Close'] = np.log(close)
            
            data['Symbol'] = 'BTC/USD'
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')
        else:
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

    elif data_name == 'crypto':
        if not os.path.exists(output_file):
            top100_csv = '/data3/finance/Top100Cryptos/100 List.csv'
            df = pd.read_csv(top100_csv, encoding='utf-8')
            top12 = df['Name'][:12].to_list()
            data = None
            for name in top12:
                print(f'crypto {name} processing...')
                csv_path = os.path.join('/data3/finance/Top100Cryptos/', name + '.csv')
                if data is not None:
                    data2 = pd.read_csv(csv_path, encoding='utf-8')
                    data2.rename(columns={'Date':'date'}, inplace=True)
                    data2['date'] = pd.to_datetime(data2.date)
                    if data2['date'].min() < pd.to_datetime('2016'): # 데이터가 너무 적은 것은 제외하기 위해 2016년 이전의 데이터가 있는 암호화폐만 선택
                        data2['Symbol'] = name
                        data2 = data2.sort_values(by=['date'])
                        data2['Market Cap'] = data2['Market Cap'].replace({'-': None})
                        data2['Market Cap'] = data2['Market Cap'].str.replace(',', '')
                        data2['Market Cap'].fillna(method='bfill', inplace=True)
                        data2['Volume'] = data2['Volume'].replace({'-': None})
                        data2['Volume'] = data2['Volume'].str.replace(',', '')
                        data2['Volume'].fillna(method='bfill', inplace=True)
                        data2['Return'] = data2['Close'].pct_change()
                        data2 = data2.dropna()
                        data = pd.concat([data, data2], axis=0)
                else:
                    data = pd.read_csv(csv_path, encoding='utf-8')
                    data.rename(columns={'Date':'date'}, inplace=True)
                    data['Symbol'] = name
                    data['date'] = pd.to_datetime(data.date)
                    data = data.sort_values(by=['date'])
                    data['Market Cap'] = data['Market Cap'].replace({'-': None})
                    data['Market Cap'] = data['Market Cap'].str.replace(',', '')
                    data['Market Cap'].fillna(method='bfill', inplace=True)
                    data['Volume'] = data['Volume'].replace({'-': None})
                    data['Volume'] = data['Volume'].str.replace(',', '')
                    data['Volume'].fillna(method='bfill', inplace=True)
                    data['Return'] = data['Close'].pct_change()
                    data = data.dropna()

            # data preprocessing
            data = data.reset_index()
            dates = pd.to_datetime(data['date'].to_list())
            data['days_from_start'] = (dates - pd.datetime(2013, 4, 28)).days
            data['day_of_week'] = dates.dayofweek.astype(str).astype('category')
            data['day_of_month'] = dates.day.astype(str).astype('category')
            data['week_of_year'] = dates.weekofyear.astype(str).astype('category')
            data['month'] = dates.month.astype(str).astype('category')
            data['year'] = dates.year.astype(str).astype('category')

            # Processes log Close
            close = data['Close'].copy()
            data['log_Close'] = np.log(close)

            # save
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')
                
        else:
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

    elif data_name == 'crypto_hourly':
        if not os.path.exists(output_file):
            filenames = ['Bitfinex_BTCUSD_1h.csv', 'Bitfinex_ETHUSD_1h.csv', 'Bitfinex_LTCUSD_1h.csv']
            data_list = []
            for f in filenames:
                data = pd.read_csv(os.path.join('/data3/finance', f), encoding='utf-8', skiprows=1)
                data.rename(columns={'Date':'date'}, inplace=True)
                data['date'] = data['date'].str.replace('-', '')
                data['date'] = pd.to_datetime(data.date)
                data = data.sort_values(by=['date'])
                data = data[['date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume USD']]
                data['Return'] = data['Close'].pct_change()
                data = data.dropna()
                data['hours_from_start'] = np.arange(len(data))
                data_list.append(data)
        
            # data preprocessing
            data = pd.concat(data_list)
            data = data.reset_index()
            dates = pd.to_datetime(data['date'].to_list())
            data['hour'] = dates.hour.astype(str).astype('category')
            data['day_of_week'] = dates.dayofweek.astype(str).astype('category')
            data['day_of_month'] = dates.day.astype(str).astype('category')
            data['week_of_year'] = dates.weekofyear.astype(str).astype('category')
            data['month'] = dates.month.astype(str).astype('category')
            data['year'] = dates.year.astype(str).astype('category')

            # Processes log Close
            close = data['Close'].copy()
            data['log_Close'] = np.log(close)

            # save
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')

        else:
            data = pd.read_csv(output_file, encoding='utf-8')
            data['hour'] = data['hour'].astype(str).astype('category')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

    elif data_name == 'crypto_daily':
        if not os.path.exists(output_file):
            filenames = ['Bitfinex_BTCUSD_d.csv', 'Bitfinex_ETHUSD_d.csv', 'Bitfinex_LTCUSD_d.csv']
            data_list = []
            for f in filenames:
                data = pd.read_csv(os.path.join('/data3/finance', f), encoding='utf-8', skiprows=1)
                data.rename(columns={'Date':'date'}, inplace=True)
                data['date'] = pd.to_datetime(data.date)
                data = data.sort_values(by=['date'])
                data = data[['date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume USD']]
                data['Return'] = data['Close'].pct_change()
                data = data.dropna()
                data['days_from_start'] = np.arange(len(data))
                data_list.append(data)
        
            # data preprocessing
            data = pd.concat(data_list)
            data = data.reset_index()
            dates = pd.to_datetime(data['date'].to_list())
            data['day_of_week'] = dates.dayofweek.astype(str).astype('category')
            data['day_of_month'] = dates.day.astype(str).astype('category')
            data['week_of_year'] = dates.weekofyear.astype(str).astype('category')
            data['month'] = dates.month.astype(str).astype('category')
            data['year'] = dates.year.astype(str).astype('category')

            # Processes log Close
            close = data['Close'].copy()
            data['log_Close'] = np.log(close)

            # save
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')

        else:
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

    elif data_name == 'sp500':
        if not os.path.exists(output_file):
            sp500_path = '/data3/finance/sp500_price'
            filenames = os.listdir(sp500_path)
            data_list = []
            for f in filenames:
                data = pd.read_csv(os.path.join(sp500_path, f), encoding='utf-8')
                data.rename(columns={'Date':'date'}, inplace=True)
                data['date'] = pd.to_datetime(data.date)
                if data['date'].min() < pd.to_datetime('2019'): # 데이터가 너무 적은 것은 제외하기 위해 2019년 이전의 데이터가 있는 종목만 선택
                    data = data.sort_values(by=['date'])
                    # data['Return'] = data['Close'].pct_change()
                    data = data.dropna()
                    data['days_from_start'] = np.arange(len(data))
                    data_list.append(data)
            print(len(data_list))
            # data preprocessing
            data = pd.concat(data_list)
            data = data.reset_index()
            dates = pd.to_datetime(data['date'].to_list())
            data['day_of_week'] = dates.dayofweek.astype(str).astype('category')
            data['day_of_month'] = dates.day.astype(str).astype('category')
            data['week_of_year'] = dates.weekofyear.astype(str).astype('category')
            data['month'] = dates.month.astype(str).astype('category')
            data['year'] = dates.year.astype(str).astype('category')

            # save
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')

        else:
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

    elif data_name == 'kospi':
        if not os.path.exists(output_file):
            kospi_path = '/data3/finance/kospi_price'
            filenames = os.listdir(kospi_path)
            data_list = []
            for f in filenames:
                data = pd.read_csv(os.path.join(kospi_path, f), encoding='utf-8')
                if len(data) == 0: # 주가 데이터가 없는 경우
                    continue
                data.rename(columns={'Date':'date'}, inplace=True)
                data['date'] = pd.to_datetime(data.date)
                if data['date'].min() < pd.to_datetime('2016'): # 데이터가 너무 적은 것은 제외하기 위해 2016년 이전의 데이터가 있는 종목만 선택
                    data = data.sort_values(by=['date'])
                    # data['Return'] = data['Close'].pct_change()
                    data = data.dropna()
                    data['days_from_start'] = np.arange(len(data)) # 주말의 공백을 없애고 주식 거래일을 연속적으로 처리
                    data_list.append(data)
            # data preprocessing
            data = pd.concat(data_list)
            data = data.reset_index()
            dates = pd.to_datetime(data['date'].to_list())
            data['day_of_week'] = dates.dayofweek.astype(str).astype('category')
            data['day_of_month'] = dates.day.astype(str).astype('category')
            data['week_of_year'] = dates.weekofyear.astype(str).astype('category')
            data['month'] = dates.month.astype(str).astype('category')
            data['year'] = dates.year.astype(str).astype('category')

            # save
            print(len(data['Symbol'].unique())) # 총 종목수
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')

        else:
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

    elif data_name == 'kospi200':
        if not os.path.exists(output_file):
            kospi = pd.read_csv('/data3/finance/kospi.csv')
            kospi200_path = '/data3/finance/kospi200_listing/'
            kospi200_list = pd.read_csv(os.path.join(kospi200_path, 'kospi200_20201218.csv'), encoding='euc-kr')
            data = kospi[kospi['Symbol'].isin(kospi200_list['종목명'])]
            data = data.set_index(data.columns[0])
            
            # data preprocessing
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

            # save
            print(len(data['Symbol'].unique())) # 총 종목수
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8')

        else:
            # preprocessed same with kospi
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

    elif data_name == 'kospi200+AUDCHF':
        if not os.path.exists(output_file):
            # load AUD/CHF data
            audchf = fdr.DataReader('AUD/CHF', '2000-01-04', '2021-02-26') # 홍콩달러/스위스프랑 환율
            audchf = audchf.reset_index()
            audchf = audchf.rename(columns={'Date':'date', 'Close': 'AUD_CHF_Close', 'Change': 'AUD_CHF_Change'})
            audchf = audchf[['date', 'AUD_CHF_Close', 'AUD_CHF_Change']]

            # load kospi200 data
            kospi200 = pd.read_csv('/data3/finance/kospi200.csv')
            kospi200['date'] = pd.to_datetime(kospi200['date'])
            
            # merge data
            data = pd.merge(kospi200, audchf, how='inner', on='date')
            data = data.sort_values(by=data.columns[0])

            # data preprocessing
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

            # save
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8', index=False)

        else:
            # preprocessed same with kospi
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

    elif data_name == 'kospi200+TI':
        '''
        ** Technical Indicators **
        - Trend: Moving Average Convergence or Divergence(MACD), Parabolic Stop And Reverse(PSAR)
        - Volatility: Bollinger Bands(BB)
        - Momentum: Stochastic Oscillator(SO), Rate Of Change(ROC)
        - Volume: On-Balance Volume(OBV), Force Index(FI)
        '''
        if not os.path.exists(output_file):
            # load kospi200 data
            data = pd.read_csv('/data3/finance/kospi200.csv')

            ## Trend indicators
            # Moving Average Convergence or Divergence(MACD)
            indicator_macd = MACD(close=data["Close"])
            data['mc_mc'] = indicator_macd.macd()
            data['mc_mcdiff'] = indicator_macd.macd_diff()
            data['mc_mcsig'] = indicator_macd.macd_signal()

            # Parabolic Stop And Reverse(PSAR)
            indicator_psar = PSARIndicator(high=data["High"], low=data["Low"], close=data["Close"])
            data['pa_pa'] = indicator_psar.psar()
            data['pa_pad'] = indicator_psar.psar_down()
            data['pa_padi'] = indicator_psar.psar_down_indicator()
            data['pa_pau'] = indicator_psar.psar_up()
            data['pa_paui'] = indicator_psar.psar_up_indicator()

            ## Volatility indicators
            # Bollinger Bands(BB)
            indicator_bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
            data['bb_bbm'] = indicator_bb.bollinger_mavg()
            data['bb_bbh'] = indicator_bb.bollinger_hband()
            data['bb_bbl'] = indicator_bb.bollinger_lband()
            data['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
            data['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

            ## Momentum indicators
            # Stochastic Oscillator(SO)
            indicator_so = StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"])
            data['so_so'] =  indicator_so.stoch()
            data['so_sosi'] = indicator_so.stoch_signal()

            # Rate Of Change(ROC)
            indicator_roc = ROCIndicator(close=data["Close"])
            data['roc'] = indicator_roc.roc()

            ## Volume indicators
            # On-Balance Volume(OBV)
            indicator_obv = OnBalanceVolumeIndicator(close=data["Close"], volume=data["Volume"])
            data['obv'] = indicator_obv.on_balance_volume()

            # Force Index(FI)
            indicator_fi = ForceIndexIndicator(close=data["Close"], volume=data["Volume"])
            data['fi'] = indicator_fi.force_index()

            # data preprocessing
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(0)

            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

            # save
            print('Completed formatting, saving to {}'.format(output_file))
            data.to_csv(output_file, encoding='utf-8', index=False)

        else:
            # preprocessed same with kospi
            data = pd.read_csv(output_file, encoding='utf-8')
            data['day_of_week'] = data['day_of_week'].astype(str).astype('category')
            data['day_of_month'] = data['day_of_month'].astype(str).astype('category')
            data['week_of_year'] = data['week_of_year'].astype(str).astype('category')
            data['month'] = data['month'].astype(str).astype('category')
            data['year'] = data['year'].astype(str).astype('category')

    return data


if __name__ == '__main__':
    pass
