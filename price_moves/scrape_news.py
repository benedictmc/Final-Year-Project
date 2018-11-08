import urllib.request
import json
import pymongo
import time 
import pandas as pd
import dateutil.parser
from datetime import datetime
from pytz import timezone
import numpy as np
import matplotlib.pyplot as plt
import binance_price_moves as binance_price
import os

class RetriveNews():
    def __init__(self):
        coin = 'STRAT'
        filename = f'data_files/{coin}/{coin}_news.csv'
        self.news_df = self.retrive_news(filename, coin)
        self.news_df = self.change_news()
        self.price_df = self.get_price(coin)
        self.price_df = self.get_price_features()
        self.price_df = self.price_fixer()
        self.data = self.combine_df(self.news_df, self.price_df)
        self.data.dropna(inplace=True)
        self.data.to_csv('data_files/STRAT/full_csv.csv')

    def get_price_features(self):
        price_df_features = pd.DataFrame(index = self.price_df.index)
        close = self.price_df.close
        volume_lengths = [3, 5, 10, 30, 60, 120, 240, 720, 1440]
        price_lengths = [2, 3, 5, 10, 15, 30, 60, 120, 240, 300, 720, 1440, 2880]
        for vol in volume_lengths:
            price_df_features[f'{vol}_volume'] = self.price_df.volume.rolling(vol).sum()
        for price in price_lengths:
            price_df_features[f'{price}_price'] = close - close.shift(price)
        price_df_features['high'] = self.price_df.high
        price_df_features['close'] = close
        return price_df_features



    def change_news(self):
        df = pd.DataFrame(index = self.news_df.index)
        last_news = 0
        time_col, count_col = [], []
        for index, row in self.news_df.iterrows():
            if row.id != 0:
                last_news = 0
                time_col.append(last_news)
            else:
                last_news+=1
                time_col.append(last_news)
        df['time_from_news'] = time_col

        for i in range(1440):
            count_col.append(None)
        for i in range(len(time_col)-1440):
            count_col.append(time_col[i:i+1440].count(0))

        df['time_from_news'] = time_col
        df['news_in_24H'] = count_col
        return df

    def combine_df(self, news_df, price_df):
        price_df.index = pd.to_datetime(price_df.index, format='%Y-%m-%d %H:%M:%S')
        news_df.index = pd.to_datetime(news_df.index, format='%Y-%m-%d %H:%M:%S')
        n_list, p_list = news_df.index.tolist(), price_df.index.tolist()
        n_first, p_first = n_list[0], p_list[0]
        n_last, p_last = n_list[-1], p_list[-1]
        min_i = max(n_first, p_first)
        max_i = min(n_last, p_last)
        news_df = news_df.truncate(before=min_i, after=max_i)
        price_df = price_df.truncate(before=min_i, after=max_i)
        self.dataset = pd.concat([news_df, price_df], axis=1, join='inner')
        return self.dataset
        
    def retrive_news(self, filename, coin):
        print('Getting news information...')
        if os.path.exists(filename):
            print('Reading from csv...')
            return pd.read_csv(filename, index_col=0 )
        url = f'https://cryptopanic.com/api/posts/?auth_token=7b50093a122a00d067104f45c8b5e0c514f45d14&currencies={coin}'
        df = pd.DataFrame(columns=['id', 'url', 'title', 'date', 'domain'])
        df = self.get_news(url, df, coin)
        news_df = self.pad_news_csv(df)
        print(f'Saving {filename} to csv..')
        news_df.to_csv(filename)
        return news_df

    def get_news(self, url, news_df, coin):
        print(f"Retriving news articles from cryptopanic for {coin}..." )
        contents = urllib.request.urlopen(url).read()
        data = json.loads(contents)
        for result in data['results']:
            if result['kind'] == 'news':
                news_obj = {'id':result['id'],'url':result['url'],'title':result['title'],'date':result['published_at'],'domain':result['domain']}
                news_df = news_df.append(news_obj, ignore_index=True)
        if data['next'] != None:
            print('Getting next page...')
            url = data['next']
            time.sleep(3)
            return self.get_news(url, news_df, coin)
        print('Done returning datframe...')
        return news_df


    def pad_news_csv(self, news_df):
        print('Padding dataframe...')
        news_df.date = [dateutil.parser.parse(i).astimezone(timezone('Europe/Dublin')) for i in news_df.date]
        news_df = news_df.set_index('date')
        news_df = news_df.resample('T').first().replace({'url':{np.nan:'-'}, 'title':{np.nan:'-'}, 'domain':{np.nan:'-'}}).fillna(0)
        print('Padding done...')
        return news_df

    def get_price(self,coin):
        print('Getting price information...')
        filename = f'data_files/{coin}/{coin}_prices.csv'
        if os.path.exists(filename):
            print('Reading from csv...')
            return pd.read_csv(filename, index_col=0 )
        else:
            print('Loading binance price file...')
            x = binance_price.PriceMoves()
            x.get_price(coin)
            return pd.read_csv(filename, index_col=0 )


    def price_fixer(self):
        three_days = 60*24*3
        rev_high = self.price_df.iloc[::-1]
        rev_high['x_range'] = rev_high.high.rolling(three_days).max()
        rev_high.dropna(inplace=True)
        rev_high = rev_high.iloc[::-1]
        rev_high['target'] = self.get_class(rev_high)
        rev_high.drop(['x_range'], axis=1, inplace = True)
        return rev_high

    def get_class(self, rev_high):
        targets = []
        for i in range(len(rev_high)):
            if rev_high.x_range.iloc[i] > rev_high.close.iloc[i]*1.15:
                targets.append(1)
            else:
                targets.append(0)
        return targets
x = RetriveNews()

