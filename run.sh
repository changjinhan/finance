### TFT ### 
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
# nohup python run.py --data=kospi200 --loss='quantile' --gpu_index=3 1>./results/tft_kospi200_epoch_50_batch_512.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='directional' --gpu_index=0 1>./results/tft_kospi200_epoch_50_batch_512_directional_01_adam.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='quantile' --gpu_index=0 1>./results/tft_kospi200_epoch_30_batch_512.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='quantile' --gpu_index=2 1>./results/tft_kospi200_epoch_50_batch_512_adam.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='directional' --gpu_index=2 1>./results/tft_kospi200_epoch_30_batch_512_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='dilate' --gpu_index=3 1>./results/tft_kospi200_pred_20_epoch_50_batch_512_dilate.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='quantile' --gpu_index=1 1>./results/tft_kospi200_pred_20_epoch_50_batch_512.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='directional' --gpu_index=2 1>./results/tft_kospi200_pred_20_epoch_50_batch_512_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='quantile' --gpu_index=1 1>./results/tft_kospi200_pred_1_epoch_50_batch_512.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='quantile' --gpu_index=0 1>./results/tft_kospi200_epoch_50_batch_256_2year.log 2>&1 &
# nohup python run.py --data=kospi200 --loss='quantile' --gpu_index=1 1>./results/tft_kospi200_epoch_50_batch_512_all_reals_added.log 2>&1 &
# nohup python run.py --data=kospi200+AUDCHF --loss='quantile' --gpu_index=2 1>./results/tft_kospi200+AUDCHF_epoch_50_batch_512.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='quantile' --gpu_index=2 1>./results/tft_kospi200+TI_epoch_50_batch_128.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='directional' --gpu_index=2 1>./results/tft_kospi200+TI_epoch_50_batch_128_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='directional' --gpu_index=3 1>./results/tft_kospi200+TI_epoch_50_batch_128_directional_05.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='dilate' --gpu_index=1 1>./results/tft_kospi200+TI_epoch_50_batch_128_dilate.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=stft --loss=quantile --gpu_index=0 1>./results/stft_entmax15_kospi200+TI_epoch_50_batch_128.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='quantile' --gpu_index=0 1>./results/tft_kospi200+TI_epoch_50_batch_128_top10.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='directional' --gpu_index=0 1>./results/tft_kospi200+TI_epoch_50_batch_128_directional_01_top10.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='dilate' --gpu_index=1 1>./results/tft_kospi200+TI_epoch_50_batch_128_dilate_top10.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=stft --loss=quantile --gpu_index=0 1>./results/stft_sparsemax_kospi200+TI_epoch_50_batch_128.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='quantile' --gpu_index=1 1>./results/tft_kospi200+TI_pred_20_epoch_50_batch_128.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='directional' --gpu_index=0 1>./results/tft_kospi200+TI_pred_20_epoch_50_batch_128_directional_01_top10.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=stft --loss=directional --gpu_index=0 1>./results/stft_entmax15_kospi200+TI_epoch_50_batch_128_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=stft --loss=directional --gpu_index=0 1>./results/stft_entmax15_kospi200+TI_epoch_50_batch_128_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='quantile' --gpu_index=3 1>./results/tft_kospi200+TI_epoch_50_batch_128_rop_5.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='directional' --gpu_index=1 1>./results/tft_kospi200+TI_epoch_50_batch_128_rop_5_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='quantile' --gpu_index=3 1>./results/tft_kospi200+TI_epoch_50_batch_128_lr_01_rop_5.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='quantile' --gpu_index=0 1>./results/tft_kospi200+TI_epoch_50_batch_128_lr_003.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='directional' --gpu_index=1 1>./results/tft_kospi200+TI_pred_20_epoch_50_batch_128_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='directional' --gpu_index=0 1>./results/tft_kospi200+TI_epoch_50_batch_128_lr_003_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='quantile' --gpu_index=3 1>./results/tft_kospi200+TI_epoch_50_batch_128_lr_005.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='directional' --gpu_index=1 1>./results/tft_kospi200+TI_epoch_50_batch_128_lr_003_directional_001.log 2>&1 &
# nohup python run.py --data=kospi200+TI --loss='quantile' --gpu_index=1 1>./results/tft_kospi200+TI_epoch_50_batch_128_lr_003_2.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=stft --loss=directional --gpu_index=1 1>./results/stft_sparsemax_kospi200+TI_epoch_50_batch_128_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=stft --loss=directional --gpu_index=0 1>./results/stft_entmax15_kospi200+TI_epoch_50_batch_128_lr_003_directional_01_predict_plot.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=stft --loss=directional --gpu_index=3 1>./results/stft_sparsemax_kospi200+TI_epoch_50_batch_128_directional_01_predict_plot.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=stft --loss=directional --gpu_index=3 1>./results/stft_sparsemax_kospi200+TI_epoch_50_batch_128_lr_003_directional_01.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=tft --loss=quantile --gpu_index=3 1>./results/tft_kospi200+TI_best_hparams.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=tft --loss=quantile --gpu_index=3 1>./results/tft_kospi200+TI_hidden_60_lr_003.log 2>&1 &
# nohup python run.py --data=kospi200+TI --model=stft --loss=directional --gpu_index=1 1>./results/stft_entmax15_kospi200+TI_epoch_50_batch_128_lr_001_directional_01_only_close.log 2>&1 &


### staged learning ###
# nohup python staged_learning.py 1>./results/tft_staged_learning_epoch5_limit_train_batch_1.log 2>&1 &
# nohup python staged_learning.py --data=kospi --gpu_index=0 1>./results/tft_staged_learning_epoch5_limit_train_batch_1_v3.log 2>&1 &
# nohup python staged_learning.py --data=kospi200+TI --loss='quantile' --gpu_index=1 1>./results/tft_staged_learning_kospi200+TI.log 2>&1 &
# nohup python staged_learning.py --data=kospi200+TI --loss='quantile' --gpu_index=0 1>./results/tft_staged_learning_kospi200+TI_rop_5.log 2>&1 &

### transfer learning ###
# nohup python run.py --data=kospi --transfer=sp500 --gpu_index=0 1>./results/tft_kospi_pretrain_sp500_epoch_100.log 2>&1 &
# nohup python run.py --data=kospi --transfer=vol --gpu_index=0 1>./results/tft_kospi_pretrain_vol_epoch_100.log 2>&1 &

### multi-GPU ###
# nohup python run.py --data=kospi --ngpu=2 --distributed_backend="ddp" 1>./results/tft_kospi_2010_multi_gpu_512_map_location.log 2>&1 &

### repeated experiment ### 
# nohup python ./repeated_experiment/quantile_vs_directional.py 1>./results/quantile_vs_directional.log 2>&1 &
# nohup python ./repeated_experiment/quantile_vs_directional.py 1>./results/quantile_vs_directional_lr_001.log 2>&1 &
# nohup python ./repeated_experiment/tft_vs_stft_entmax15.py 1>./results/tft_vs_stft_entmax15.log 2>&1 &
# nohup python ./repeated_experiment/stft_directional.py 1>./results/stft_entmax15_directional.log 2>&1 &
# nohup python ./repeated_experiment/stft_directional.py 1>./results/stft_sparsemax_directional.log 2>&1 &

### Auto-ARIMA###  
# nohup python run_arima.py 1>./results/arima_kospi_2010.log 2>&1 &
# nohup python run_arima.py 1>./results/arima_kospi200.log 2>&1 &

### DeepAR ### 
# nohup python run_deepar.py --data=vol --ngpu=0 1>./results/deepar_vol_epoch_1_batch_512.log 2>&1 &
# nohup python run_deepar.py --data=kospi200 --gpu_index=0 1>./results/deepar_kospi200_epoch_50_batch_128.log 2>&1 &
# nohup python run_deepar.py --data=kospi200 --gpu_index=2 1>./results/deepar_kospi200_pred20_epoch_50_batch_128.log 2>&1 &

### MQRNN ###
# nohup python run_mqrnn.py 1>./results/mqrnn_kospi200_lr003.log 2>&1 &

### Seq2Seq ###
# nohup python run_seq2seq.py --data=kospi200+TI --gpu_index=0 1>./results/seq2seq_kospi200+TI_1.log 2>&1 &
# nohup python run_seq2seq.py --data=kospi200+TI --gpu_index=1 1>./results/seq2seq_kospi200+TI_2.log 2>&1 &

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
# nohup python hyperparameter_tuning.py --data=kospi200+TI --gpu_index=2 1>./results/tft_kospi200+TI_op.log 2>&1 &

### Backtest ###
# nohup python backtest.py --data=kospi200+TI --loss='quantile' --strategy='SMA' 1>./results/backtest_kospi200+TI_quantile_SMA.log 2>&1 &
# nohup python backtest.py --data=kospi200+TI --loss='directional(0.1)' --strategy='SMA' 1>./results/backtest_kospi200+TI_directional_01_SMA.log 2>&1 &
# nohup python backtest.py --data=kospi200+TI --loss='dilate' --strategy='SMA' 1>./results/backtest_kospi200+TI_dilate_SMA.log 2>&1 &
# nohup python backtest.py --data=kospi200+TI --loss='dilate' --strategy='TFT' 1>./results/backtest_kospi200+TI_dilate.log 2>&1 &
# nohup python backtest.py --data=kospi200+TI --model='entmax15' --loss='directional(0.1)' --strategy='SMA' 1>./results/backtest_kospi200+TI_entmax15_directional_01_SMA.log 2>&1 &
# nohup python backtest.py --data=kospi200+TI --model='entmax15' --loss='directional(0.1)' --strategy='TFT' 1>./results/backtest_kospi200+TI_entmax15_directional_01.log 2>&1 &
