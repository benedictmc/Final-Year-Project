import binance_dataset_update as binance
import pandas as pd
# import rnn_price_class as price_class
import OHLC_preprocessing as make_features
import real_time_OHLC as rt_make_features
import use_model as run_model

class AppDriver():
    def __init__(self):
        mode, window, pair = 'japan',  'minute', 'BTCUSDT'
        coin = pair[:3]
        if mode == 'japan':
            filename = f'data_files/training/japan_btc.csv'
            print('Starting OHLCPreprocess')
            x = make_features.OHLCPreprocess(filename, coin, window)
            print('Starting PriceClassification')
            price_class.PriceClassification(x.save_filename)
        if mode  == 'train':
            print(f'Starting App Driver for training on {coin} {window} data')
            binance.BinanceDS('update', window, pair)
            filename = f'data_files/{window}/master_dataset_{coin}.csv'
            print('Starting OHLCPreprocess')
            x = make_features.OHLCPreprocess(filename, coin, window)
            print('Starting PriceClassification')
            price_class.PriceClassification(x.save_filename)
        elif mode == 'real_time':
            print(f'Starting App Driver for real time mode on {coin} {window} data')
            real_time = rt_make_features.OHLCRealTime(pair, window)
            print(f'Starting the use model file....')
            run_model.ActualPrediction(real_time.all_data)







        
x = AppDriver()