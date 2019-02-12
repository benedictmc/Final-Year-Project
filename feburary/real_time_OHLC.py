import pandas as pd 
import numpy as np
import plotly as py
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from datetime import datetime
import scipy.optimize
import warnings
import matplotlib.pyplot as plt
import math
import talib
from scipy import stats
import json
import binance_dataset_update as Binance


class OHLCRealTime():
    def __init__(self, realtime=False, coin = ''):
        self.METHOD_CALLS = {
            'moving_averages' : self.moving_averages,
            "rsi": self.rsi,
            'stochastic': self.stochastic,
            'macd' : self.macd,
            'williams': self.williams,
            "bollinger_bands" : self.bollinger_bands,
        }
        ##Initialse dataframe from master BTC datafile. Datafile is in minute time periods
        print('Starting OHLCRealTime script...')
        print('Calling Binance script...')
        filename = 'data_files/minute/master_dataset_WAV.csv'
        print(coin)
        if realtime:
            filename = f'data_files/real_time/{coin}.csv'
        self.data = pd.read_csv(filename, index_col = 0)

        # ##Reduces row size to 200 for time sake        
        # self.data.index = [datetime.fromtimestamp(int(x)).strftime('%d.%m.%Y %H:%M:%S') for x in self.data.index]
        # ##Creates an index variable as this will be used often
        self.index =  self.data.index


        self.all_data = self.combine_indicators() 
        self.all_data.dropna(inplace=True)

        ## Saves to CSV
        # self.all_data.to_csv('BTC_combined.csv')



    def combine_indicators(self):
        all_df = pd.DataFrame(index=self.data.index)
        all_df['close'] = self.data.close
        all_df['volume'] = self.data.volume
        with open('metadata/period_lists.json', 'r') as f:
            data_list = json.load(f)

        for method in self.METHOD_CALLS:
            period_list = data_list[method]
            function = self.METHOD_CALLS[method]
            if len(period_list) != 0:
                for period in period_list:
                    print('Calling method {} for a period of {}'.format(method, str(period)))
                    df = function(period)
                    if all_df.empty:
                        all_df = df
                    else:
                        all_df = pd.concat([all_df, df], axis=1, join='inner')
            else: 
                print('Calling method {} for a period of {}'.format(method, str(period)))
                if all_df.empty:
                    all_df = function()
                else:
                    all_df = pd.concat([all_df, function()], axis=1, join='inner')
        all_df.dropna(inplace=True)
        return all_df


    def moving_averages(self, period):
        df = pd.DataFrame(index=self.index)
        close = self.data.close 
        df[f'ma_{period}'] = close.rolling(period).mean()
        return df       
        
    def rsi(self, period):
        df = pd.DataFrame(index=self.index)
        close = self.data.close 
        rsi = talib.RSI(close, timeperiod=14)    
        df[f'rsi_{period}'] = rsi
        return df

    def stochastic(self, period):
        df = pd.DataFrame(index=self.index)
        high, low, close = self.data.high, self.data.low, self.data.close
        slowk, slowd = talib.STOCH(high, 
                            low, 
                            close, 
                            fastk_period=period, 
                            slowk_period=3, 
                            slowk_matype=0, 
                            slowd_period=3, 
                            slowd_matype=0)
        if period == 9:
            df['9_kstochastic'] = slowk
            df['9_dstochastic'] = slowd
        else:
            ##Returning the difference between K line and D line. This means closer to zero means a crossover
            df['stochastic_{}'.format(str(period))] = (slowk - slowd)
        return df

    def momentum(self, period):
        df = pd.DataFrame(index=self.index)
        close = self.data.close
        momentum = talib.MOM(close, timeperiod=period)
        df['momentum_{}'.format(str(period))] = momentum
        return df
            

    def williams(self, period):
        df = pd.DataFrame(index=self.index)
        high, low, close = self.data.high, self.data.low, self.data.close
        williams = talib.WILLR(high, low, close, timeperiod=period)
        df['williams_{}'.format(str(period))] = williams
        return df

    def PROC(self):
        df = pd.DataFrame(index=self.index)
        close = self.data.close
        percent_change = (close - close.shift(1))/close.shift(1)
        df['%_change'] = percent_change
        return df

    def get_percentage_change(self, values):
        return (values-values.shift(1))/values.shift(1) 
   
    def macd(self):
        df = pd.DataFrame(index=self.index)
        close = self.data.close
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=15, slowperiod=30, signalperiod=9)
        df['macd_hist'] = macdhist
        return df


    def cci(self):
        df = pd.DataFrame(index=self.index)
        high, low, close = self.data.high, self.data.low, self.data.close
        cci = talib.CCI(high, low, close, timeperiod=14)
        df['cci'] = cci
        return df

    def bollinger_bands(self):
        df = pd.DataFrame(index=self.index)
        close = self.data.close
        upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=15, nbdevup=2, nbdevdn=2, matype=0)
        df['%b_bands'], df['midlle_bands'] = ((close-lowerband)/(upperband-lowerband)), middleband
        return df


    def build_classification(self):
        print("Getting target classification information")
        self.all_data['future'] = self.all_data.close.shift(-3)
        self.all_data['target'] = list(map(self.get_class, self.all_data.close,  self.all_data['future']))
        self.all_data = self.all_data.drop('future', 1)

    def get_class(self, current, future):
        if float(current)< float(future) :
            return 1
        else:
            return 0

    def preprocess(self, df):
        minmax_scale = preprocessing.MinMaxScaler().fit(df)
        df_minmax = minmax_scale.transform(df)
        return df_minmax
