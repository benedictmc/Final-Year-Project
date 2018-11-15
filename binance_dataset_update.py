import os
from binance.client import Client
import pandas as pd
from datetime import datetime
import json
import logging
import time

class BinanceDS():  
    FILENAME = ''
    API = ''
    API_SECRET = ''
    COIN_CONTEXT = ''
    PERIOD = 3
    # PAIR_LIST = ['BTCUSDT']
    MINUTES = 100000
    PAIR_LIST = ['BTCUSDT','ETHUSDT', 'EOSUSDT', 'XRPUSDT']
    X,y = '', ''

    def __init__(self, time):
        # logging.basicConfig(filename='binance.log',level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        with open('metadata/binance_keys.txt') as f:
            keys = f.read()
            keys = keys.split(',')
            BinanceDS.API, BinanceDS.API_SECRET = keys[0], keys[1]
        self.client = Client(BinanceDS.API, BinanceDS.API_SECRET)
        self.logger.info("Starting to retrive dataset.....")
        if time == 'hour':
            self.update_datasets_hour()
        else:
            self.update_datasets()
    def update_datasets(self):
        print('Starting to update master datasets...')
        column_list = ['date','Open','High','Low','Close','Volume']
        for coin_pair in BinanceDS.PAIR_LIST:
            update = False
            df = pd.DataFrame()
            coin = coin_pair[:3]
            print('Updating {} dataset...'.format(coin))
            filelist = os.listdir('dataset_files/master/')
            print("Getting pricing information for {}".format(coin))
            filename = 'dataset_files/master/master_dataset_{}.csv'.format(coin)
            if os.path.isfile(filename):
                update = True
                read_df = pd.read_csv(filename, index_col= 0)
                last_index = read_df.index[-1]
                time_s = int(time.time())
                reminder = time_s % 60
                time_s = time_s - reminder
                time_diff = int((time_s - last_index)/60)
                print('Master file found for coin {} updating file with {} minutes of data'.format(coin, time_diff))
            elif filelist != []:
                file = filelist[0]
                read_df = pd.read_csv('dataset_files/master/'+ file, index_col= 0)
                first_index = read_df.index[0]
                time_s = int(time.time())
                reminder = time_s % 60
                time_s = time_s - reminder
                time_diff = int((time_s - first_index)/60)
                print('Master file not found for coin {} updating file with {} minutes of data from file {}'.format(coin, time_diff, file))
            else:
                read_df = pd.DataFrame()
                time_diff = BinanceDS.MINUTES
                print('No master files found setting up file for coins with data from {} minutes ago'.format(BinanceDS.MINUTES))
            print('Done')
            
            df = pd.DataFrame()
            price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1MINUTE, "{} minutes ago GMT".format(time_diff))
            for index, col in enumerate(column_list):
                if col == 'date':
                    df['date'] = [int(entry[0])/1000 for entry in price_data]
                    continue

                df[col] = [entry[index] for entry in price_data]
                df[col] = df[col].astype('float64')

            df.set_index('date', inplace=True)
            if update:
                df = pd.concat([read_df, df])
            df.to_csv('dataset_files/master/master_dataset_{}.csv'.format(coin))
    
    def update_datasets_hour(self):
        print('Starting to update master datasets...')
        column_list = ['date','Open','High','Low','Close','Volume']
        for coin_pair in BinanceDS.PAIR_LIST:
            update = False
            coin = coin_pair[:3]
            print('Updating {} dataset...'.format(coin))
            filelist = os.listdir('dataset_files/master/hour')
            print("Getting pricing information for {}".format(coin))
            filename = 'dataset_files/master/hour/master_dataset_{}.csv'.format(coin)
            if os.path.isfile(filename):
                update = True
                read_df = pd.read_csv(filename, index_col= 0)
                last_index = read_df.index[-1]
                time_s = int(time.time())
                reminder = time_s % 60
                time_s = time_s - reminder
                time_diff = int((time_s - last_index)/60)
                print('Master file found for coin {} updating file with {} hours of data'.format(coin, time_diff))
            elif filelist != []:
                file = filelist[0]
                read_df = pd.read_csv('dataset_files/master/hour/'+ file, index_col= 0)
                first_index = read_df.index[0]
                time_s = int(time.time())
                reminder = time_s % 60
                time_s = time_s - reminder
                time_diff = int((time_s - first_index)/60)
                print('Master file not found for coin {} updating file with {} hours of data from file {}'.format(coin, time_diff, file))
            else:
                read_df = pd.DataFrame()
                time_diff = BinanceDS.MINUTES
                print('No master files found setting up file for coins with data from {} hours ago'.format(BinanceDS.MINUTES))
            print('Done')
            
            df = pd.DataFrame()
            price_data = self.client.get_historical_klines(coin_pair, Client.KLINE_INTERVAL_1HOUR, "{} hour ago GMT".format(time_diff))
            for index, col in enumerate(column_list):
                if col == 'date':
                    df['date'] = [int(entry[0])/1000 for entry in price_data]
                    continue

                df[col] = [entry[index] for entry in price_data]
                df[col] = df[col].astype('float64')

            df.set_index('date', inplace=True)
            if update:
                df = pd.concat([read_df, df])
            df.to_csv('dataset_files/master/hour/master_dataset_{}.csv'.format(coin))
