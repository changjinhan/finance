import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset


def read_df(config:dict):
    """
    This function is for reading the sample testing dataframe
    """
    symbol = config['symbol']
    df = pd.read_csv('/data3/finance/kospi200.csv')
    df = df[df['Symbol'] == symbol]

    # preprocessing
    df['day_of_week'] = df['day_of_week'].astype(int)
    df['day_of_month'] = df['day_of_month'].astype(int)
    df['week_of_year'] = df['week_of_year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['year'] = df['year'].astype(int)
    df['Close'] = df['Close'].astype(int)

    target_df = pd.DataFrame(index=df['days_from_start'], 
                             data={'close': df['Close'].to_numpy()})
    horizon_size = config['horizon_size']
    covariate_df = pd.DataFrame(index=target_df.index,
                                data={
                                    'dayofweek': df['day_of_week'].to_numpy(),
                                    'dayofmonth': df['day_of_month'].to_numpy(),
                                    'weekofyear': df['week_of_year'].to_numpy(),
                                    'month': df['month'].to_numpy(),
                                    'year': df['year'].to_numpy()
                                    })
    for col in covariate_df.columns:
        covariate_df[col] = (covariate_df[col] - np.mean(covariate_df[col]))/np.std(covariate_df[col])
    train_target_df = target_df.iloc[:-horizon_size,:]
    test_target_df = target_df.iloc[-horizon_size:,:]
    train_covariate_df = covariate_df.iloc[:-horizon_size,:]
    test_covariate_df = covariate_df.iloc[-horizon_size:,:]

    # small_train_target_df = train_target_df.iloc[-1000:,:].copy()
    # small_train_covariate_df = train_covariate_df.iloc[-1000:,:].copy()
    return train_target_df, test_target_df, train_covariate_df, test_covariate_df


class MQRNN_dataset(Dataset):
    
    def __init__(self,
                series_df:pd.DataFrame,
                covariate_df:pd.DataFrame, 
                horizon_size:int,
                quantile_size:int):
        
        self.series_df = series_df
        self.covariate_df = covariate_df
        self.horizon_size = horizon_size
        self.quantile_size = quantile_size
        full_covariate = []
        covariate_size = self.covariate_df.shape[1]
        print(f"self.covariate_df.shape[0] : {self.covariate_df.shape[0]}")
        for i in range(1, self.covariate_df.shape[0] - horizon_size+1):
            cur_covariate = []
            #for j in range(horizon_size):
            cur_covariate.append(self.covariate_df.iloc[i:i+horizon_size,:].to_numpy())
            full_covariate.append(cur_covariate)
        full_covariate = np.array(full_covariate)
        print(f"full_covariate shape: {full_covariate.shape}")
        full_covariate = full_covariate.reshape(-1, horizon_size * covariate_size)
        self.next_covariate = full_covariate
    
    def __len__(self):
        return self.series_df.shape[1]
    
    def __getitem__(self,idx):
        cur_series = np.array(self.series_df.iloc[: -self.horizon_size, idx])
        cur_covariate = np.array(self.covariate_df.iloc[:-self.horizon_size, :]) # covariate used in generating hidden states

        covariate_size = self.covariate_df.shape[1]
        #next_covariate = np.array(self.covariate_df.iloc[1:-self.horizon_size+1,:]) # covariate used in the MLP decoders

        real_vals_list = []
        for i in range(1, self.horizon_size+1):
            real_vals_list.append(np.array(self.series_df.iloc[i: self.series_df.shape[0]-self.horizon_size+i, idx]))
        real_vals_array = np.array(real_vals_list) #[horizon_size, seq_len]
        real_vals_array = real_vals_array.T #[seq_len, horizon_size]
        cur_series_tensor = torch.tensor(cur_series)
        
        cur_series_tensor = torch.unsqueeze(cur_series_tensor,dim=1) # [seq_len, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate) #[seq_len, covariate_size]
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor],dim=1)
        next_covariate_tensor = torch.tensor(self.next_covariate) #[seq_len, horizon_size * covariate_size]

        cur_real_vals_tensor = torch.tensor(real_vals_array)
        return cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor


