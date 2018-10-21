from tensorflow.keras.models import load_model
import pandas as pd
from binance.client import Client
import Trading_OHLC_preprocessing as Trading
import binance_dataset_update as Update
import numpy as np 
from collections import deque
import time
from datetime import datetime

class TradingSimulator():
    SEQ_LENGTH = 15 
    API, API_SECRET = '', ''


    def __init__(self, coin):
        print('Starting trading simulator for coin {}...'.format(coin))
        with open('metadata/binance_keys.txt') as f:
            keys = f.read()
            keys = keys.split(',')
            TradingSimulator.API, TradingSimulator.API_SECRET = keys[0], keys[1]
        # self.client = Client(TradingSimulator.API, TradingSimulator.API_SECRET)
        self.filepath ='models/RNN_Final-12-0.548.model'
        self.coin = coin
        # Update.BinanceDS('minute')
        ds_filepath = 'dataset_files/master/master_dataset_{}.csv'.format(coin)
        df = pd.read_csv(ds_filepath, index_col=0)
        self.dataset = df.tail(300)
        post_process = Trading.OHLCPreprocess(self.dataset)


        self.market_data = post_process.post_process_df
        sequential_data, index_list = self.build_sequences(self.market_data)
        self.X, self.y = self.extract_feature_labels(sequential_data)

        self.x_df = pd.DataFrame(columns=['x','y','close'], index=index_list)
        close = post_process.close_data
        for i in range(0, len(self.x_df)):
            self.x_df.iloc[i].x = self.X[i]
            self.x_df.iloc[i].y = self.y[i]
            self.x_df.iloc[i].close = close.iloc[i]
        print(self.x_df.close) 
        balance, crypto_cal = self.simulate_trading(self.filepath)
        print('Simulation ended: End balance is {} and crypto balance is {}'.format(balance, crypto_cal))

        # self.start_real_time()


    def verify_y(self):
        furture_close = self.x_df.close.shift(-3)
        for i in range(0, len(self.x_df)-3):
            if self.x_df.close.iloc[i] < furture_close[i]:
                print('Current and Future was {} and {}'.format(self.x_df.close.iloc[i]*1.00055, furture_close[i]))
                print(self.x_df.y.iloc[i])
            else:
                print('y was {}'.format(self.x_df.y.iloc[i]))
    
    def build_sequences(self, df):
        sequential_data, index_list = [], []
        prev_days = deque(maxlen = TradingSimulator.SEQ_LENGTH)
        for index, row in df.iterrows():
            prev_days.append([n for n in row[:-1]])
            if len(prev_days) == TradingSimulator.SEQ_LENGTH:
                sequential_data.append([np.array(prev_days), row.iloc[-1]])
                index_list.append(index)
        return sequential_data, index_list

    def extract_feature_labels(self, seq_data):
        X, y = [], []
        for seq, target in seq_data:
            X.append(seq)
            y.append(target)
        return np.array(X), y

    def simulate_trading(self, filepath):
        self.model = load_model(filepath)
        predicted_y = self.model.predict_classes(self.X, batch_size = 128)
        correct, false = 0,0
        balance, crypto_bal, bought, sold = 500, 0, False, True
        for prediction, y, close in zip(predicted_y, self.x_df.y.values, self.x_df.close.values):
            if prediction == 1 and not bought:
                crypto_bal = balance/close
                bought, sold = True, False
                balance = 0
                print('Buying: Price {} Cryto Balance {}'.format(close, crypto_bal))
                print('Predicted y: {}, Actual y: {}'.format(prediction, y))
                print('****************')
            if prediction == 1 and bought:
                print('Holding: Price {} still holding'.format(close))
            if prediction == 0 and bought:
                balance = crypto_bal*close
                bought, sold = False, True
                crypto_bal = 0
                print('Selling: Price {} Balance {}'.format(close, balance))
                print('Predicted y: {}, Actual y: {}'.format(prediction, y))
                print('****************')
            if prediction == 0 and sold:
                print('Waiting: Price {} still holding'.format(close))
            if prediction == y:
                correct +=1
            else:
                false +=1

        print('There was {} amount of correct predictions, and {} amount of false predictions'.format(correct, false))
        return balance, crypto_bal


    def start_real_time(self):
        from twisted.internet import task
        from twisted.internet import reactor
        timeout = 1.0
        # hist_data =(int(time.time() - int(time.time()))%60) - 60*30
        # print()
        def start():
            if int(time.time())%60 == 0:
                print("Start")
                

        # l = task.LoopingCall(start)
        # l.start(timeout)

        # reactor.run()

        def get_past_data():
            df = pd.DataFrame()
            column_list = ['date','Open','High','Low','Close','Volume']
            df = pd.DataFrame()
            price_data = self.client.get_historical_klines('XRPUSDT', Client.KLINE_INTERVAL_1MINUTE, "{} minutes ago GMT".format(89))
            for index, col in enumerate(column_list):
                if col == 'date':
                    time = [int(entry[0])/1000 for entry in price_data]
                    df['date'] = time
                    continue
                df[col] = [entry[index] for entry in price_data]
                df[col] = df[col].astype('float64')
            df.set_index('date', inplace=True)

            x = Indicators.MarketIndicators(df, 'XRP', False, True)

            # print(df.index)
            print(x.indicator_df.index[-1])

            # self.model = load_model(self.filepath)

            X = build_sequences(x.indicator_df)
            print(len(X))
            # predicted_y = self.model.predict_classes(X, batch_size = 64)
            # print(predicted_y)

        def build_sequences(df):
            sequential_data = []
            prev_days = deque(maxlen = 60)
            for i in df.values:
                prev_days.append([n for n in i[:-1]])
                if len(prev_days) == 60:
                    sequential_data.append(np.array(prev_days))
            
            return np.array(sequential_data)
        

x = TradingSimulator('BTC')