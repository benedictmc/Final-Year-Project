from binance.client import Client
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

class PriceMoves():
    API, API_SECRET = '', ''

    def __init__(self):
        with open('../metadata/binance_keys.txt') as f:
            keys = f.read()
            keys = keys.split(',')
            PriceMoves.API, PriceMoves.API_SECRET = keys[0], keys[1]
        self.client = Client(PriceMoves.API, PriceMoves.API_SECRET)
        self.pairs = []
        tickers = self.client.get_orderbook_tickers()

        for ticker in tickers:
           self.pairs.append(ticker['symbol'])


        self.plot_move('Salt Coin')
        # self.load_close()
        # self.google_values()

    def get_prices(self):
        column_list = ['date','Open','High','Low','Close','Volume']
        for pair in self.pairs:
            print(f'Getting price data for {pair}')
            df = pd.DataFrame()
            price_data = self.client.get_historical_klines(pair, Client.KLINE_INTERVAL_1DAY, "30 day ago GMT")
            for index, col in enumerate(column_list):
                if col == 'date':
                    df['date'] = [int(entry[0])/1000 for entry in price_data]
                    continue
                df[col] = [entry[index] for entry in price_data]
                df[col] = df[col].astype('float64')
            print('Finished. Saving to csv')
            df.to_csv(f'../dataset_files/price_moves/{pair}.csv')


    def load_close(self):
        big_diff = 0
        diff_l = {}
        directory = '../dataset_files/price_moves/'
        price_csvs = os.listdir(directory)
        for file in price_csvs:
            df = pd.read_csv(directory+file, index_col=0)
            diff = (df.Close - df.Close.shift(1) )/df.Close.shift(1)
            self.plot_move(diff.values,file)
            diff = diff.max()
            diff_l[file] = diff
        for key, value in diff_l.items():
            print(f'Pair of {key} had a biggest value of {value}')
            

    def google_values(self):
        from pytrends.request import TrendReq
        kw_list = ["salt coin"]
        pytrends = TrendReq(hl='en-US', tz=360)
        x = pytrends.get_historical_interest(kw_list, year_start=2018, month_start=10, day_start=1, year_end=2018, month_end=10, day_end=20, sleep=0)
        x.to_csv(f'../dataset_files/google_trends/{kw_list[0]}.csv')


    def plot_move(self, coinpair):
        min_max_scaler = preprocessing.MinMaxScaler()

        df_google = pd.read_csv(f'../dataset_files/google_trends/salt coin.csv', index_col=0)
        df_google.index = pd.to_datetime(df_google.index, format='%Y-%m-%d %H:%M:%S')
        df_google = df_google.resample('24H').mean()


        df_google = df_google.replace([np.inf, -np.inf], np.nan)
        df_google.dropna(inplace=True)
        df_google = df_google[['salt coin']]
        print(df_google)
        y2 = min_max_scaler.fit_transform(df_google.values)


        df_price = pd.read_csv('../dataset_files/price_moves/SALTBTC.csv', index_col=0)
        # df_price['diffr'] = (df_price.Close - df_price.Close.shift(1) )/df_price.Close.shift(1)
        # df_price['diffr'] = df_price['diffr'].tail(len(y2))


        df_price['Close'] = df_price['Close'].tail(len(y2))
        df_price = df_price.replace([np.inf, -np.inf], np.nan)
        df_price.dropna(inplace=True)
        df_price = df_price[['Close']] 
        y = min_max_scaler.fit_transform(df_price.values)



        x = np.arange(len(y))


        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, y2, marker='', color='olive', linewidth=2)


        ax.set(xlabel='Days (d)', ylabel='Price Change (%)',
            title=f'Percent change over 30 days for {coinpair}')
        ax.grid()

        plt.show()

x = PriceMoves()