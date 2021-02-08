### Experiment with each dataset ###
# nohup python run.py --data=vol --gpu_index=0 1>./results/tft_vol_3months.log 2>&1 &
# nohup python run_nbeats.py --data=vol --gpu_index=0 1>./results/nbeats_vol_3months.log 2>&1 &
# nohup python run.py --data=stock_idx --gpu_index=0 1>tft_stock_idx.log 2>&1 &
# nohup python run.py --data=btc_krw --gpu_index=0 1>tft_btc_krw.log 2>&1 &
# nohup python run.py --data=btc_usd --gpu_index=0 1>tft_btc_usd.log 2>&1 &
# nohup python run.py --data=crypto --gpu_index=0 1>tft_crypto.log 2>&1 &
# nohup python run.py --data=crypto_daily --gpu_index=0 1>tft_crypto_daily.log 2>&1 &
# nohup python run.py --data=crypto_hourly --gpu_index=0 1>tft_crypto_hourly.log 2>&1 &
# nohup python run.py --data=sp500 --gpu_index=0 1>./results/tft_sp500_epoch_100_2.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=0 1>./results/tft_kospi_epoch_100_sparsemax.log 2>&1 &
# nohup python run.py --data=kospi --ngpu=0 1>./results/tft_kospi_epoch_100_upgrade_visualizing.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=0 1>./results/tft_kospi_batch_512.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=1 1>./results/tft_kospi_directional_quantile_loss.log 2>&1 &
nohup python run.py --data=kospi --gpu_index=1 1>./results/tft_kospi_epoch_1000_patience_50_directional_alpha.log 2>&1 &


# nohup python run.py --data=kospi --transfer=sp500 --gpu_index=0 1>./results/tft_kospi_pretrain_sp500_epoch_100.log 2>&1 &
# nohup python run.py --data=kospi --transfer=vol --gpu_index=0 1>./results/tft_kospi_pretrain_vol_epoch_100.log 2>&1 &


### hyper parmameter tuning ###
# nohup python hyperparameter_tuning.py --data=vol --gpu_index=1 1>tft_vol_op.log 2>&1 &
# nohup python hypterparameter_tuning.py --data=stock_idx --gpu_index=1 1>tft_stock_idx_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=btc_krw --gpu_index=1 1>tft_btc_krw_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=btc_usd --gpu_index=1 1>tft_btc_usd_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=crypto --gpu_index=1 1>tft_crypto_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=crypto_daily --gpu_index=1 1>tft_crypto_daily_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=crypto_hourly --gpu_index=1 1>tft_crypto_hourly_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=sp500 --gpu_index=0 1>./results/tft_sp500_op.log 2>&1 &
# nohup python hyperparameter_tuning.py --data=kospi --gpu_index=1 1>./results/tft_kospi_op.log 2>&1 &
