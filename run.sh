### Experiment with each dataset ###
# nohup python run.py --data=vol --gpu_index=0 1>./results/tft_vol_3months.log 2>&1 &
# nohup python run_nbeats.py --data=vol --gpu_index=0 1>./results/nbeats_vol_3months.log 2>&1 &
# nohup python run.py --data=stock_idx --gpu_index=0 1>tft_stock_idx.log 2>&1 &
# nohup python run.py --data=btc_krw --gpu_index=0 1>tft_btc_krw.log 2>&1 &
# nohup python run.py --data=btc_usd --gpu_index=0 1>tft_btc_usd.log 2>&1 &
# nohup python run.py --data=crypto --gpu_index=0 1>tft_crypto.log 2>&1 &
# nohup python run.py --data=crypto_daily --gpu_index=0 1>tft_crypto_daily.log 2>&1 &
# nohup python run.py --data=crypto_hourly --gpu_index=0 1>tft_crypto_hourly.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=0 1>./results/tft_kospi_epoch_100_sparsemax.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=0 1>./results/tft_kospi_batch_512.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=0 1>./results/tft_kospi_epoch_50_overfit_check.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=3 1>./results/tft_kospi_2010_epoch_50_overfit_check_batch512_clip_0.01.log 2>&1 &
# nohup python run.py --data=vol --gpu_index=1 1>./results/tft_vol_dilate_quantile.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=1 1>./results/tft_kospi_2010_dilate_quantile_epoch_500_patience_100.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=1 1>./results/tft_kospi_2010_DILATE_output_minmax.log 2>&1 &
# nohup python run.py --data=vol --gpu_index=0 1>./results/tft_vol_DILATE_output_minmax.log 2>&1 &
# nohup python run.py --data=kospi --gpu_index=1 1>./results/tft_kospi_2010_dilate_quantile_epoch_500_patience_100_weight_001.log 2>&1 &
# nohup python run.py --data=kospi --loss='directional' --gpu_index=2 1>./results/tft_kospi_2010_epoch_20_batch_512_directional_weight_01.log 2>&1 &
# nohup python run.py --data=kospi --loss='quantile' --gpu_index=1 1>./results/tft_kospi_2010_epoch_50_batch_512_minmax.log 2>&1 &
# nohup python run.py --data=kospi --loss='dilate' --gpu_index=0 1>./results/tft_kospi_2010_epoch_20_batch_512_dilatequantile_weight_001.log 2>&1 &
# nohup python run.py --data=kospi --loss='quantile' --gpu_index=0 1>./results/tft_new_kospi_2010_epoch_20_batch_512.log 2>&1 &
nohup python run.py --data=kospi200 --loss='quantile' --gpu_index=3 1>./results/tft_kospi200_epoch_50_batch_512.log 2>&1 &

## staged learning
# nohup python staged_learning.py 1>./results/tft_staged_learning_epoch5_limit_train_batch_1.log 2>&1 &
# nohup python staged_learning.py --data=kospi --gpu_index=0 1>./results/tft_staged_learning_epoch5_limit_train_batch_1_v3.log 2>&1 &

## transfer learning
# nohup python run.py --data=kospi --transfer=sp500 --gpu_index=0 1>./results/tft_kospi_pretrain_sp500_epoch_100.log 2>&1 &
# nohup python run.py --data=kospi --transfer=vol --gpu_index=0 1>./results/tft_kospi_pretrain_vol_epoch_100.log 2>&1 &

## multi-GPU
# nohup python run.py --data=kospi --ngpu=2 --distributed_backend="ddp" 1>./results/tft_kospi_2010_multi_gpu_512_map_location.log 2>&1 &

### repeated experiment ###
# nohup python ./repeated_experiment/quantile_vs_directional.py 1>./results/quantile_vs_directional.log 2>&1 &
# nohup python ./repeated_experiment/quantile_vs_directional.py 1>./results/quantile_vs_directional_512.log 2>&1 &

### Auto-ARIMA ###
# nohup python run_arima.py 1>./results/arima_kospi_2010.log 2>&1 &

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
