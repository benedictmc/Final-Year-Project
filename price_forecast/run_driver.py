import binance_dataset_update as binance
import pandas as pd
import rnn_price_class as price_class

import OHLC_preprocessing as make_features
class AppDriver():
    def __init__(self):
        train, window, pair = True,  'minute', 'BTCUSDT'
        coin = pair[:3]
        if train:
            print(f'Starting App Driver for training on {coin} {window} data')
            binance.BinanceDS('update', window, pair)
            filename = f'data_files/{window}/master_dataset_{coin}.csv'
            print('Starting OHLCPreprocess')
            x = make_features.OHLCPreprocess(filename, coin, window)
            print('Starting PriceClassification')
            price_class.PriceClassification(x.save_filename)
            

        
x = AppDriver()