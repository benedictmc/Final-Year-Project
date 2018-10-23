from binance.client import Client
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import talib
from pytrends.request import TrendReq

class Features():
    API, API_SECRET = '', ''
    FEATURE_LIST = {
        "price_change" : ['1','2','4','6','12','24','48','96']
    }

    def __init__(self):
        with open('../metadata/binance_keys.txt') as f:
            keys = f.read()
            keys = keys.split(',')
            Features.API, Features.API_SECRET = keys[0], keys[1]
        self.client = Client(Features.API, Features.API_SECRET)
        # filename = '../dataset_files/price_moves/hour/ADABNB.csv'
        # self.data = pd.read_csv(filename, index_col=0)
        # self.close = self.data.Close
        # features_df = pd.DataFrame()
        # self.get_price_change()
        # self.read_google_trends()
        print(self.client.get_all_tickers())

    def get_price_change(self):
        df = pd.DataFrame()
        df['close'] = self.close        
        for change in Features.FEATURE_LIST['price_change']:
            change = int(change)
            if int(change) / 24 >= 1:
                name = change / 24
                name = str(int(name))+'D'
            else:
                name = str(change)+'H'
            df[f'{name}_change'] = (self.close-self.close.shift(change)) / self.close.shift(change)
        print(df)


    def big_price_move(self):
        pass


    def read_google_trends(self):
        google_df = pd.read_csv('dock_trends_hour.csv', index_col=0)
        google_df['trends'] = google_df['dock coin: (Worldwide)']
        google_df= google_df.drop('dock coin: (Worldwide)', axis=1)
        google_df.index = google_df.index.str.replace('T', ' ')

        google_df.index = pd.to_datetime(google_df.index, infer_datetime_format=True)
        ema_trends = talib.EMA(google_df.trends, timeperiod=5)
        price_df = self.get_price_temp()
        price_df.index = pd.to_datetime(price_df.index, infer_datetime_format=True, unit='s')

        print(google_df.index)
        print(price_df.index)
        for i ,ix in zip(google_df.index, price_df.index):
            print(f'Google index {i} price index {ix}')



        # # + pd.DateOffset(minutes=4)
        # trun_time = google_df.index[0] 
        # print(google_df.index[0])
        # price_df = price_df.truncate(before = trun_time)
        # price_df= price_df.resample('8T').sum()
        # print(price_df.index)
        # print(ema_trends.index)


        # y2 = min_max_scaler.fit_transform(df_google.values)


        fig, ax1 = plt.subplots()
        t = price_df.index
        s1 = price_df.close
        ax1.plot(t, s1, 'b-')
        ax1.set_xlabel('Date hour (h)')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('Close prices', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        s2 = ema_trends
        ax2.plot(t, s2, 'r-')
        ax2.set_ylabel('Google Trends 7 EMA', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()
        plt.show()


    def get_price_temp(self):
        pair = 'DOCKBTC'
        if os.path.exists(f'{pair}_hour.csv'):
            df = pd.read_csv(f'{pair}_hour.csv', index_col=0)
            return df
        else:
            df = pd.DataFrame()
            print(f'Getting price data for {pair}')
            column_list = ['date','open','high','low','close','volume']
            price_data = self.client.get_historical_klines(pair, Client.KLINE_INTERVAL_1HOUR, "7 day ago UTC")
            for index, col in enumerate(column_list):
                if col == 'date':
                    df['date'] = [int(entry[0]/1000)+3600 for entry in price_data]
                    continue
                df[col] = [entry[index] for entry in price_data]
                df[col] = df[col].astype('float64')
            df.set_index('date', inplace=True)
            print('Finished. Saving to csv')
            df.to_csv(f'{pair}_hour.csv')
            return df




x = Features()