import os
import warnings
import numpy as np
import pandas as pd
import copy
import FinanceDataReader as fdr

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
                if data['date'].min() < pd.to_datetime('2018'): # 데이터가 너무 적은 것은 제외하기 위해 2018년 이전의 데이터가 있는 종목만 선택
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

    return data


if __name__ == '__main__':
    pass
