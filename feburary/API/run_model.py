from tensorflow.keras.models import load_model
import pandas as pd
from collections import deque
import numpy as np
import binance_dataset_update as update_df
import real_time_OHLC as OHLC_data
from sklearn import preprocessing
import json

def fetch_data(update = False, realtime = False):
    prediction_coin = 'BTC'
    pair_list, coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT'] , ['BTC', "ETH", 'BNB', 'XRP']
    if update:
        for pair in pair_list:
            update_df.BinanceDS('update', 'minute', pair)
    if realtime:
        print('Updating datasets for realtime.. ')
        for pair in pair_list:
            update_df.BinanceDS('real', 'minute', pair)

    data = pd.DataFrame()
    unprocessed_close_vals = ''
    if realtime:
        for coin in coins:
            filepath = f'data_files/real_time/{coin}.csv'
            df = pd.read_csv(filepath, index_col = 0)
            df = df[['close', 'volume']]
            if coin == prediction_coin:
                unprocessed_close_vals = df[['close']]
                unprocessed_close_vals['time'] = df.index
                
            df.columns = [f'close_{coin}', f'volume_{coin}']
            if data.empty:
                data = df
            else:
                data = data.join(df)
        OHLC_obj = OHLC_data.OHLCRealTime(realtime= True, coin='BTC')
        df = OHLC_obj.all_data
        data = data.join(df)
        data.dropna(inplace=True)
        return data , unprocessed_close_vals
    else:
        for coin in coins:
            filepath = f'data_files/minute/master_dataset_{coin}.csv'
            df = pd.read_csv(filepath, names=['time','open','high','low','close','volume'], index_col = 0)
            df = df[['close', 'volume']]
            df.columns = [f'close_{coin}', f'volume_{coin}']

            if data.empty:
                data = df
            else:
                data = data.join(df)

def get_percentage_change(values):
    return (values-values.shift(1))/values.shift(1) 
    
def preprocess(df):
    for col in df.columns:
        if col != 'target':
            df[col] = get_percentage_change(df[col])
            df = df.replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df


def use_model(x_seq, filename = ''):
    filepath = '../models/MODEL SEQ_LEN60_PRED_1549365876.2177374'
    model = load_model(filepath)
    predicted_price = model.predict_classes(x_seq)
    return predicted_price


def run_profit_loss(predicted_price, close_vals, time = ''):
    bal, bought, sold = 6300, False, True
    for i in range(0, len(predicted_price)):
        if predicted_price[i] == 1 and sold:
            bal = bal / close_vals[i]
            bought, sold = True, False
            print(f'BUYING: {close_vals[i]} at time {time[i]}')
        if predicted_price[i] == 0  and bought:
            bal = bal * close_vals[i]
            bought, sold = False, True
            print(f'SELLING: {close_vals[i]} at time {time[i]}')
    if bought:
        bal = bal * close_vals[-1]
    print(f'Final Balance is {bal} at time {time[-1]}')


def build_sequences(df, length):
    sequential_data = []
    close_vals = []
    prev_days = deque(maxlen = length)
    for i in df.values:
        prev_days.append([n for n in i[:-2]])
        if len(prev_days) == length:
            close_vals.append(i[-1])
            sequential_data.append([np.array(prev_days), i[-2]])
    return extract_feature_labels(sequential_data, close_vals)


def build_realtime_squences(df, length):
    sequential_data = []
    close_vals = []
    time = []
    prev_days = deque(maxlen = length)
    for i in df.values:
        prev_days.append([n for n in i[:-2]])
        if len(prev_days) == length:
            close_vals.append(i[-2])
            time.append(i[-1])
            sequential_data.append(np.array(prev_days))
    return np.array(sequential_data), close_vals, time

def extract_feature_labels(seq_data, close_vals):
    X, y = [], []
    for seq, target in seq_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y, close_vals


def update_fetaures(columns):
    with open('data/metadata.json', 'r') as f:
        metadata = json.load(f)

    metadata['features'] = columns

    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f)

def run(mode, prediction_coin, update_features=False, run_prediction = False):
    if mode == 'static':
        df = pd.read_csv('validation.csv', index_col =0 )
        df_all = pd.read_csv('unprocessed.csv', index_col =0 )
        df['BTC_close_up'] = df_all[['close_BTC']]

        X, y, close_vals = build_sequences(df, 16)
        predicted = use_model(X)
        run_profit_loss(predicted, close_vals)

    if mode == 'realtime':
        data, unprocessed_close_vals = fetch_data(realtime=True)
        if update_features:
            update_fetaures(list(data.columns))
        data.to_csv(f'data_files/visual/{prediction_coin}_raw.csv')   
        if run_prediction:    
            data = preprocess(data)

            data[f'{prediction_coin}_close_up'] = unprocessed_close_vals['close']
            data[f'time'] = unprocessed_close_vals['time']

            X, close_vals, time = build_realtime_squences(data, 16)
            df_result = pd.DataFrame(index=time)

            predicted = use_model(X)
            df_result['signal'] = predicted
            df_result.to_csv('data_files/visual/BTC_signals.csv')        
        # run_profit_loss(predicted, close_vals, time)

def run_signals(coin):
    data, unprocessed_close_vals = fetch_data(realtime=True)
    data = preprocess(data)

    data[f'{coin}_close_up'] = unprocessed_close_vals['close']
    data[f'time'] = unprocessed_close_vals['time']

    X, close_vals, time = build_realtime_squences(data, 16)
    df_result = pd.DataFrame(index=time)

    predicted = use_model(X)
    df_result['signal'] = predicted
    df_result['close'] = unprocessed_close_vals['close']

    df_result.to_csv('data_files/visual/BTC_signals.csv')  
    

def test_close():
    df = pd.read_csv('data_files/visual/BTC_signals.csv')
    print(df.values[0][1:])



mode  = 'realtime'
prediction_coin = 'BTC'
# run_signals(prediction_coin)
