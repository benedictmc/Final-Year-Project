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
    SEQ_LENGTH = 60 
    API, API_SECRET = '', ''


    def __init__(self, coin):
        print('Starting trading simulator for coin {}...'.format(coin))
        with open('metadata/binance_keys.txt') as f:
            keys = f.read()
            keys = keys.split(',')
            TradingSimulator.API, TradingSimulator.API_SECRET = keys[0], keys[1]
        self.client = Client(TradingSimulator.API, TradingSimulator.API_SECRET)
        self.filepath ='models/RNN_Final-05-0.881.model'
        self.coin = coin
        Update.BinanceDS()
        ds_filepath = 'dataset_files/master/master_dataset_{}.csv'.format(coin)
        df = pd.read_csv(ds_filepath, index_col=0)
        self.dataset = df.tail(3000)
        post_process = Trading.OHLCPreprocess(self.dataset)
        self.market_data = post_process.all_data
        # print(post_process.close_data)
        self.X, index_list = self.build_sequences(self.market_data)
        # self.find_close_test()
        self.x_df = pd.DataFrame(columns=['x', 'close'], index=index_list)
        for i in range(0, len(self.x_df)):
            self.x_df.iloc[i].x = self.X[i]
        self.x_df.close = post_process.close_data
        balance, crypto_cal = self.simulate_trading(self.filepath)
        print('Simulation ended: End balance is {} and crypto balance is {}'.format(balance, crypto_cal))

        # self.start_real_time()


    def build_sequences(self, df):
        sequential_data, index_list = [], []
        prev_days = deque(maxlen = TradingSimulator.SEQ_LENGTH)
        for index, row in df.iterrows():
            prev_days.append([n for n in row])
            if len(prev_days) == TradingSimulator.SEQ_LENGTH:
                sequential_data.append(np.array(prev_days))
                index_list.append(index)
        return np.array(sequential_data), index_list

    def simulate_trading(self, filepath):
        self.model = load_model(filepath)
        predicted_y = self.model.predict_classes(self.X, batch_size = 128)
    
        balance, crypto_bal, bought, sold = 500, 0, False, True
        for prediction, close in zip(predicted_y, self.x_df.close.values):
            if prediction == 1 and not bought:
                crypto_bal = balance/close
                bought, sold = True, False
                print('Buying: Price {} Cryto Balance {}'.format(close, crypto_bal))
            if prediction == 1 and bought:
                print('Holding: Price {} still holding'.format(close))
            if prediction == 0 and bought:
                balance = crypto_bal*close
                bought, sold = False, True
                print('Selling: Price {} Balance {}'.format(close, balance))
            if prediction == 0 and sold:
                print('Waiting: Price {} still holding'.format(close))


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