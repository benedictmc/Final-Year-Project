import os
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import numpy as np 
from tensorflow.keras.models import load_model
from collections import deque
import random 
import time
import real_time_OHLC

class ActualPrediction():

    def __init__(self):
        print("Starting price classification script ...")
        filename = 'post.csv'
        print('Starting use_model script...')
        print('Calling real_timeOHLC script...')

        real_time = real_time_OHLC.OHLCRealTime()
        self.dataset = real_time.all_data
        print('Done calling scripts.... ')

        diff_list = ['ma_5','ma_6','ma_10','%b_bands','midlle_bands','rsi_6','rsi_12','williams_12','9_kstochastic','9_dstochastic','macd_hist']
        change_df = pd.DataFrame()
        for col in diff_list:
            if col != 'close':
                change_df[f'%_{col}'] = self.get_percentage_change(self.dataset[col])

        self.dataset = pd.concat([self.dataset, change_df], axis=1, join='inner')
        self.dataset = self.dataset.replace([np.inf, -np.inf], np.nan)
        self.dataset.dropna(inplace=True)

        self.dataset.drop(diff_list, inplace=True, axis=1)
        cols = list(self.dataset.columns.values) 
        cols.pop(cols.index('close')) 

        self.dataset = self.dataset[cols+['close']]
        target = self.dataset.close.shift(-3)
        
        
        self.dataset.dropna(inplace=True)
        self.dataset.index = pd.to_datetime(self.dataset.index, format="%d.%m.%Y %H:%M:%S")


        self.scaler = preprocessing.MinMaxScaler()
        self.post_df = self.preprocess(self.dataset, self.scaler)


        train_seq = self.build_sequences(self.post_df)
        X_seq, y = self.extract_feature_labels(train_seq)
        self.use_model(X_seq[-1], y[-1])

        # df = pd.read_csv(filename, index_col=0)
        # df = self.buy_sell(df)
        # self.run_profit_loss(df)

    def preprocess(self, df, scaler):
        # s = preprocessing.MinMaxScaler().fit(df.values)
        df_minmax = scaler.fit_transform(df.values)
        return df_minmax

    def buy_sell(self, df):
        df['versus'] = df.actual.shift(3)
        df['buy_sell'] = [1 if row[1].predicted >= row[1].versus else 0 for row in df.iterrows()]
        return df
        
    def run_profit_loss(self, df):
        bal, bought, sold = 500, False, True
        for row in df.iterrows():
            if row[1].buy_sell == 1 and sold:
                bal = bal / row[1].versus
                bought, sold = True, False
                print(f'BUYING: {bal} ')
            if row[1].buy_sell == 0 and bought:
                bal = bal * row[1].versus
                bought, sold = False, True
                print(f'SELLING: {bal} ')
        print(f'Final Balance is {bal}')

    def build_sequences(self, df):
        SEQ_LENGTH = 5
        sequential_data = []
        prev_days = deque(maxlen = SEQ_LENGTH)
        for i in df:
            prev_days.append([n for n in i[:-1]])
            if len(prev_days) == SEQ_LENGTH:
                sequential_data.append([np.array(prev_days), i[-1]])
        return sequential_data


    def extract_feature_labels(self, seq_data):
        X, y = [], []
        for seq, target in seq_data:
            X.append(seq)
            y.append(target)
        return np.array(X), y

    def get_percentage_change(self, values):
        return (values-values.shift(1))/values.shift(1) 

    def use_model(self, x_seq, y):
        model = load_model('../models/RNN_Final-02-0.003.model')
        predicted_price = model.predict(x_seq)
        print(predicted_price)
        print(y)
        
x = ActualPrediction()