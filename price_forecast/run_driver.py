import binance_dataset_update as binance
import pandas as pd
# import rnn_price_class as price_class
import OHLC_preprocessing as make_features
import real_time_OHLC as rt_make_features
import use_model as run_model

class AppDriver():
    def __init__(self):
        mode, window, pair = 'train',  'hour', 'OAXBTC'
        coin = pair[:3]
        if mode == 'japan':
            print('Running train on Japan data..')
            filename = 'data_files/training/japan_btc_post.csv' 
            x = make_features.OHLCPreprocess(filename, coin, window)
            print('Starting PriceClassification')
            price_class.PriceClassification(x.save_filename, 10, coin)

        if mode  == 'train':
            print(f'Starting App Driver for training on {coin} {window} data')
            binance.BinanceDS('update', window, pair)
            filename = f'data_files/{window}/master_dataset_{coin}.csv'
            print('Starting OHLCPreprocess')
            x = make_features.OHLCPreprocess(filename, coin, window)
            print('Starting PriceClassification')
            price_class.PriceClassification(x.save_filename, 10, coin)

            
        elif mode == 'real':
            print(f'Starting App Driver for real time mode on {coin} {window} data')
            real_time = rt_make_features.OHLCRealTime(pair, window)
            print(f'Starting the use model file....')
            run_model.ActualPrediction(real_time.all_data)







        
x = AppDriver()