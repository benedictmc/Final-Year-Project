import os
from binance.client import Client
import pandas as pd
from datetime import datetime
import json
import logging

class BinanceDS():
    FILENAME = 'dataset_files/dataset_{}.csv'.format(datetime.now().strftime("%Y-%m-%d"))
    API = ''
    API_SECRET = ''
    DAYS = 300
    X,y = '', ''

    def __init__(self):
        logging.basicConfig(filename='binance.log',level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        with open('binance_keys.txt') as f:
            keys = f.read()
            keys = keys.split(',')
            BinanceDS.API, BinanceDS.API_SECRET = keys[0], keys[1]
        self.client = Client(BinanceDS.API, BinanceDS.API_SECRET)
        self.feature_cols = []
        if os.path.exists(BinanceDS.FILENAME):
            print("Using existing dataset")
            self.logger.info("Using existing dataset")
            self.dataset = pd.read_csv(BinanceDS.FILENAME)
            print(self.dataset)
            self.dataset = self.dataset.sort_values(by=['date'])
            self.get_feature_cols()
            BinanceDS.X, BinanceDS.y = self.finialise_dataset()

        else:
            self.logger.info("Creating new dataset")
            print("Creating new dataset")
            self.dataset = pd.DataFrame()
            with open('dataset_files/coin_pair.json') as f:
                coin_pairs = json.load(f)
            for coin in coin_pairs:
                print('Getting price information for '+ coin)
                self.get_price_info(coin)
            self.dataset = self.dataset.sort_values(by=['date'])
            BinanceDS.X, BinanceDS.y = self.finialise_dataset()


    def finialise_dataset(self):
        self.dataset.fillna(0, inplace=True)
        df_vals = self.dataset[[*self.feature_cols]]
        self.dataset.to_csv(BinanceDS.FILENAME, index=False)
        X = df_vals.values
        y = self.dataset['target'].values
        return X, y

    def get_price_info(self, coin_pair):
        df = pd.DataFrame()
        coin = coin_pair[:3]
        price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1DAY, "{} day ago GMT".format(BinanceDS.DAYS))
        time = [entry[0] for entry in price_data]
        df['date'] = [datetime.fromtimestamp(int(x)/1000).strftime('%Y-%m-%d') for x in time]
        df['close'] = [entry[4] for entry in price_data]
        df['coin'] = coin
        df = self.rolling_change(coin, df)
        if self.dataset.empty:
            self.dataset = df
        else:
            self.dataset = pd.concat([self.dataset, df], ignore_index=True)

    def build_features_labels(self, df):
        print("Building classification for row, rate is currently at {}".format('0.07'))
        df['target'] = list(map(self.build_classification, *self.feature_list))
        df['target'] = df['target'].shift(-1)
        return df

    def rolling_change(self, coin, df, period = 7):
        ## Converting close price to float 
        price_data = self.client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1DAY, "{} day ago GMT".format(BinanceDS.DAYS))
        self.feature_list = []
        df['close'] = df['close'].astype(float)
        print("Getting the rolling change for {} days".format(str(period)))
        for i in range(1, period+1):
            self.feature_cols.append('price_{}d'.format(i))
            df['price_{}d'.format(i)] = (df['close'] - df['close'].shift(i)) / df['close']
            self.feature_list.append(df['price_{}d'.format(i)])
        print("Adding the BTC diff two day to row")
        df['btc_diff'] = [entry[4] for entry in price_data]
        df['btc_diff'] = df['btc_diff'].astype(float)
        df['btc_diff'] = (df['btc_diff'] - df['btc_diff'].shift(1)) / df['btc_diff']
        return self.build_features_labels(df)

    def build_classification(self, *args):
        requirement = 0.07
        cols = [c for c in args]
        for col in cols:
            if col > requirement:
                return 1
            else:
                return 0

    def get_feature_cols(self, period = 7):
            for i in range(1, period+1):self.feature_cols.append('price_{}d'.format(i))

x = BinanceDS()