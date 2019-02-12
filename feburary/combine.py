import numpy as np
import binance_dataset_update as update_df
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque
import random
import tensorflow as tf 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization, Embedding
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
from tensorflow.keras.models import load_model
import json
import real_time_OHLC as OHLC_data

pair_list, coins = ['WAVESBTC', 'ETHBTC', 'BNBBTC', 'XRPBTC'] , ['WAV', "ETH", 'BNB', 'XRP']
# pair_list = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
# coins = ['BTC', "ETH", 'BNB', 'XRP']

# data = pd.read_csv('data_files/static/combined.csv', index_col=0)
result_save = {}

def build_classification(data, coin):
    print("Getting target classification information")
    coin_values = data[f'close_{coin}']
    data['future'] = coin_values.shift(-3)
    data['target'] = list(map(get_class, coin_values,  data['future']))
    data = data.drop('future', 1)
    return data

def get_class(current, future):
    fee_1, fee_2 = current*0.00075, future*0.00075
    if float(current)+ fee_1 + fee_2 < float(future) :
        return 1
    else:
        return 0

def preprocess(df):
    print(df.head())
    for col in df.columns:
        if col != 'target':
            df[col] = get_percentage_change(df[col])
            df = df.replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace=True)
            print(df[col])
            df[col] = preprocessing.scale(df[col].values)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df

def balance_data(sequential_data):
    buys, sells = [], []
    for seq, target in sequential_data:
        if target == 1:
            buys.append([seq, target])
        else:
            sells.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    buys, sells = buys[:lower], sells[:lower]
    sequential_data = buys + sells
    print('The length of buys is {}, The length of sells is {}'.format(len(buys), len(sells)))

    random.shuffle(sequential_data)
    return sequential_data

def build_sequences(df, length):
    sequential_data = []
    prev_days = deque(maxlen = length)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == length:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)
    return balance_data(sequential_data)


def extract_feature_labels(seq_data):
    X, y = [], []
    for seq, target in seq_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y


def build_model(X_train, X_val, y_train, y_val, size, epochs):
    name = 'MODEL SEQ_LEN{}_PRED_{}'.format(str(60), str(time.time()))
    model = Sequential()

    model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
    checkpoint = ModelCheckpoint("../models/{}.model".format(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')) 

    history = model.fit(X_train, y_train,
                    batch_size=size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=[tensorboard, checkpoint])

    # Score model
    score = model.evaluate(X_val, y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save("models/{}".format(name))
    return score[0], score[1]

def get_percentage_change(values):
    return (values-values.shift(1))/values.shift(1) 

def fetch_data(update = False):
    prediction_coin = 'WAV'

    if update:
        for pair in pair_list:
            update_df.BinanceDS('update', 'minute', pair)

    data = pd.DataFrame()

    for coin in coins:
        filepath = f'data_files/minute/master_dataset_{coin}.csv'
        df = pd.read_csv(filepath, index_col = 0)
        df = df[['close', 'volume']]
        df.columns = [f'close_{coin}', f'volume_{coin}']

        if data.empty:
            data = df
        else:
            data = data.join(df)

    OHLC_obj = OHLC_data.OHLCRealTime(realtime= False, coin=prediction_coin)
    df = OHLC_obj.all_data
    data = data.join(df)
    data.dropna(inplace=True)
    return data



def use_model(x_seq, filename = ''):
    filepath = 'models/MODEL SEQ_LEN60_PRED_1549299509.4460618'
    model = load_model(filepath)
    predicted_price = model.predict(x_seq)
    return predicted_price


##Config
sequence_length = 16
batch_size = 32
epochs = 15
coin = 'WAV'


##Running program
data = fetch_data(update=False)
data = build_classification(data, coin)
train_df, validation_df = train_test_split(data, train_size=0.95, shuffle= False)
validation_df.to_csv('unprocessed.csv')
print(train_df.head())
train_df = preprocess(train_df)
validation_df = preprocess(validation_df)


print("\n\n********")
print(f'Building model with squence length of {sequence_length}, batch size of {batch_size}, and epochs of {epochs}')
print("********\n\n")

sequences = build_sequences(train_df, sequence_length)
sequences_val = build_sequences(validation_df, sequence_length)

X_train, y_train = extract_feature_labels(sequences)
X_validation, y_validation = extract_feature_labels(sequences_val)
loss, acc = build_model(X_train, X_validation, y_train, y_validation, batch_size, epochs)


result_save[time.time()] = {
    'Predicting Coin' : coin,
    'sequence_length' :sequence_length,
    'batch size': batch_size,
    'epochs': epochs,
    'results' : {
        'loss' : loss,
        'accuracy' : acc
    }
}
print(result_save)
with open(f'result_{time.time()}.json', 'w') as f:
    json.dump(result_save, f)

