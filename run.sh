nohup python run.py --data=vol --gpu_index=0 1>tft_vol.log 2>&1 &
nohup python run.py --data=btc_krw --gpu_index=0 1>tft_btc_krw.log 2>&1 &
nohup python run.py --data=btc_usd --gpu_index=0 1>tft_btc_usd.log 2>&1 &
nohup python run.py --data=crypto --gpu_index=0 1>tft_crypto.log 2>&1 &