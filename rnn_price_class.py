import os
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import numpy as np 
from collections import deque
import random 
import time
import tensorflow as tf 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

class PriceClassification():
    SEQ_LENGTH = 60 
    PERIOD = 3
    EPOCHS = 15
    BATCH_SIZE = 64
    NAME = 'MODEL SEQ_LEN{}_PRED_{}'.format(str(60), str(time.time()))

    def __init__(self, filename, coin_context):
        # datetime.fromtimestamp(int(x)/1000).strftime('%Y-%m-%d') for x in time
        if os.path.exists(filename):
            print("Reading in dataset")
            self.coin_context = coin_context
            self.dataset = pd.read_csv(filename, index_col=0)
            self.dataset = self.dataset.drop('future', 1)
            self.validation_df, self.train_df = self.get_fivepct()

            # Building Training Data
            self.train_df = self.preprocess(self.train_df)
            train_seq = self.build_sequences(self.train_df)
            X_train, y_train = self.extract_feature_labels(train_seq)
            # Building Validation Data
            self.validation_df = self.preprocess(self.validation_df)
            validate_seq = self.build_sequences(self.validation_df)
            X_val, y_val = self.extract_feature_labels(validate_seq)
            self.print_statistics(X_train, X_val, y_train, y_val)
            self.build_model(X_train, X_val, y_train, y_val)
        else:
            print("Dataset {} not found. Program exiting".format(filename))

    def get_fivepct(self):
        times = sorted(self.dataset.index.values)
        last_5pct = times[-int(len(times)*(0.05))]
        validation_df = self.dataset[(self.dataset.index >= last_5pct)]
        train_df = self.dataset[(self.dataset.index < last_5pct)]
        return validation_df, train_df


    def preprocess(self, df):
        for col in df.columns:
            if col != 'target':
                df[col] = df[col].pct_change()
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna()
                df[col] = preprocessing.scale(df[col].values)
                df.dropna(inplace=True)
        return df


    def build_sequences(self, df):
        sequential_data = []
        prev_days = deque(maxlen = PriceClassification.SEQ_LENGTH)
        for i in df.values:
            prev_days.append([n for n in i[:-1]])
            if len(prev_days) == PriceClassification.SEQ_LENGTH:
                sequential_data.append([np.array(prev_days), i[-1]])
        random.shuffle(sequential_data)
        return self.balance_data(sequential_data)

    def balance_data(self, sequential_data):
        buys, sells = [], []
        for seq, target in sequential_data:
            if target == 1:
                buys.append([seq, target])
            else:
                sells.append([seq, target])
        lower = min(len(buys), len(sells))

        buys, sells = buys[:lower], sells[:lower] 
        sequential_data = buys + sells
        random.shuffle(sequential_data)

        return sequential_data

    def extract_feature_labels(self, seq_data):
        X, y = [], []
        for seq, target in seq_data:
            X.append(seq)
            y.append(target)
        return np.array(X), y

    def print_statistics(self, X_train, X_val, y_train, y_val):
        print("Train Data Length: {}, Validatation Data Length: {}".format(len(X_train), len(X_val) ))
        print("TRAIN: Amount of don't Buys: {}, Amount of Buys: {}".format(y_train.count(0), y_train.count(1)))
        print("VALIDATE: Amount of don't Buys: {}, Amount of Buys: {}".format(y_val.count(0), y_val.count(1)))



    def build_model(self, X_train, X_val, y_train, y_val):
        model = Sequential()

        model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:])))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='softmax'))

        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        tensorboard = TensorBoard(log_dir='logs/{}'.format(PriceClassification.NAME))

        filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}_{}"
        checkpoint = ModelCheckpoint("models_xrp/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) 

        history = model.fit(X_train, y_train,
                        batch_size=PriceClassification.BATCH_SIZE,
                        epochs=PriceClassification.EPOCHS,
                        validation_data=(X_val, y_val),
                        callbacks=[tensorboard, checkpoint])


    
x = PriceClassification("dataset_files/dataset_20-15-02_XRP_min.csv", "XRP")