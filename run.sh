### Experiment with each dataset ###
# nohup python run.py --data=vol --gpu_index=0 1>./results/tft_vol_3months.log 2>&1 &
# nohup python run_nbeats.py --data=vol --gpu_index=0 1>./results/nbeats_vol_3months.log 2>&1 &
# nohup python run.py --data=stock_idx --gpu_index=0 1>tft_stock_idx.log 2>&1 &
# nohup python run.py --data=btc_krw --gpu_index=0 1>tft_btc_krw.log 2>&1 &
# nohup python run.py --data=btc_usd --gpu_index=0 1>tft_btc_usd.log 2>&1 &
# nohup python run.py --data=crypto --gpu_index=0 1>tft_crypto.log 2>&1 &
# nohup python run.py --data=crypto_daily --gpu_index=0 1>tft_crypto_daily.log 2>&1 &
# nohup python run.py --data=crypto_hourly --gpu_index=0 1>tft_crypto_hourly.log 2>&1 &
# nohup python run.py --data=sp500 --gpu_index=0 1>./results/tft_sp500_sector_added.log 2>&1 &
nohup python run.py --data=kospi --transfer=sp500 --gpu_index=0 1>./results/tft_kospi_pretrain.log 2>&1 &


### hyper parmameter tuning ###
# nohup python hyperparameter_tuning.py --data=vol --gpu_index=1 1>tft_vol_op.log 2>&1 &
# nohup python hypterparameter_tuning.py --data=stock_idx --gpu_index=1 1>tft_stock_idx_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=btc_krw --gpu_index=1 1>tft_btc_krw_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=btc_usd --gpu_index=1 1>tft_btc_usd_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=crypto --gpu_index=1 1>tft_crypto_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=crypto_daily --gpu_index=1 1>tft_crypto_daily_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=crypto_hourly --gpu_index=1 1>tft_crypto_hourly_op.log 2>&1 &
