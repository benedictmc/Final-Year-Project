import os
from binance.client import Client
import pandas as pd
from datetime import datetime
import json
import logging
import time as t

class BinanceDS():  
    FILENAME = ''
    API = ''
    API_SECRET = ''
    COIN_CONTEXT = ''
    PERIOD = 3
    # PAIR_LIST = ['BTCUSDT']
    MINUTES = 100000
    PAIR_LIST = ['BCCBTC']
    X,y = '', ''

    def __init__(self, function, time, pair):
        # logging.basicConfig(filename='binance.log',level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        with open('metadata/binance_keys.txt') as f:
            keys = f.read()
            keys = keys.split(',')
            BinanceDS.API, BinanceDS.API_SECRET = keys[0], keys[1]
        self.client = Client(BinanceDS.API, BinanceDS.API_SECRET)
        self.logger.info("Starting to retrive dataset.....")
        if function == 'update':
            if time == 'minute':
                self.update_minute(time, pair)
            if time == 'hour':
                self.update_datasets(time, pair)
            if time == 'day':
                self.update_datasets(time, pair)
        elif function == 'real':
            self.real_data = self.get_real_time_data(pair, time)

    def update_datasets(self, time, coin_pair):
        print('Starting to update master datasets...')
        column_list = ['date','open','high','low','close','volume']
        df = pd.DataFrame()
        coin = coin_pair[:3]
        filelist = os.listdir(f'data_files/{time}/')
        print("Getting pricing information for {}".format(coin))
        print(f"Time selected is {time}".format(coin))

        filename = f'data_files/{time}/master_dataset_{coin}.csv'
        time_diff = 100000
        df = pd.DataFrame()
        if time =='minute':
            price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1MINUTE, "{} minutes ago GMT".format(time_diff))
        if time =='hour':
            price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1HOUR, "{} hours ago GMT".format(time_diff))
        if time =='day':
            price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1DAY, "{} days ago GMT".format(time_diff))

        for index, col in enumerate(column_list):
            if col == 'date':
                df['date'] = [int(entry[0])/1000 for entry in price_data]
                continue

            df[col] = [entry[index] for entry in price_data]
            df[col] = df[col].astype('float64')

        df.set_index('date', inplace=True)
        df.to_csv(filename)
        print('Done')
    
    def get_real_time_data(self, coin_pair, time):
            df = pd.DataFrame()
            coin = coin_pair[:3]
            print("Getting pricing information for {}".format(coin))
            print(f"Time selected is {time}".format(coin))
            column_list = ['date','open','high','low','close','volume']
            filename = f'data_files/real_time/{coin}.csv'
            df = pd.DataFrame()
            time_diff = 100
            if time =='minute':
                price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1MINUTE, "{} minutes ago GMT".format(time_diff))
            if time =='hour':
                price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1HOUR, "{} hours ago GMT".format(time_diff))
            if time =='day':
                price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1DAY, "{} days ago GMT".format(time_diff))

            for index, col in enumerate(column_list):
                if col == 'date':
                    df['date'] = [int(entry[0])/1000 for entry in price_data]
                    continue

                df[col] = [entry[index] for entry in price_data]
                df[col] = df[col].astype('float64')

            df.set_index('date', inplace=True)
            df.to_csv(filename)
            print('Done')
            return df

    def update_minute(self, time, coin_pair):
        print('Starting to update master datasets...')
        column_list = ['date','open','high','low','close','volume']
        update = False
        coin = coin_pair[:3]
        print('Updating {} dataset...'.format(coin))
        print("Getting pricing information for {}".format(coin))
        filename = f'data_files/{time}/master_dataset_{coin}.csv'
        if os.path.isfile(filename):
            update = True
            read_df = pd.read_csv(filename, index_col= 0)
            last_index = read_df.index[-1]
            time_s = int(t.time())
            reminder = time_s % 60
            time_s = time_s - reminder
            time_diff = int((time_s - last_index)/60)
            print('Master file found for coin {} updating file with {} minutes of data'.format(coin, time_diff))
        else:
            read_df = pd.DataFrame()
            time_diff = BinanceDS.MINUTES
            print('No master files found setting up file for coins with data from {} minutes ago'.format(BinanceDS.MINUTES))
        
        df = pd.DataFrame()
        if time =='minute':
            price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1MINUTE, "{} minutes ago GMT".format(time_diff))
        if time =='hour':
            price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1HOUR, "{} hours ago GMT".format(time_diff))
        if time =='day':
            price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1DAY, "{} days ago GMT".format(time_diff))
        for index, col in enumerate(column_list):
            if col == 'date':
                df['date'] = [int(entry[0])/1000 for entry in price_data]
                continue

            df[col] = [entry[index] for entry in price_data]
            df[col] = df[col].astype('float64')
        df.set_index('date', inplace=True)
        if update:
            df = pd.concat([read_df, df])
        df.to_csv(filename)
        print('Done')
        