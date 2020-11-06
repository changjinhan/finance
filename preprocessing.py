import os
import warnings
import numpy as np
import pandas as pd
import copy
import FinanceDataReader as fdr

warnings.filterwarnings("ignore")


def preprocess(data_name):
    output_file = os.path.join('/data3/finance/', data_name + '.csv')

    if data_name == 'volatility':
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
            '.N225': 'APAC ',
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
        
    elif data_name == 'bitcoin':
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

    return data


if __name__ == '__main__':
    pass
