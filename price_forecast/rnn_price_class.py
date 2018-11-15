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
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization, Embedding, Activation, GRU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt



class PriceClassification():
    SEQ_LENGTH = 5
    PERIOD = 3
    EPOCHS = 60
    BATCH_SIZE = 128
    NAME = 'MODEL SEQ_LEN{}_PRED_{}'.format(str(60), str(time.time()))

    def __init__(self, train_file):
        print("Starting price classification script ...")
        filename = train_file
        if os.path.exists(filename):
            print("Reading in dataset with filename {}".format(filename))
            self.dataset = pd.read_csv(filename, index_col=0)
            
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
            self.dataset.close = self.dataset.close.shift(-3)
            self.dataset.dropna(inplace=True)
            self.dataset.index = pd.to_datetime(self.dataset.index, format="%d.%m.%Y %H:%M:%S")

            self.validation_df, self.train_df = self.get_fivepct()
            self.close_price = self.validation_df.iloc[4:].close
            self.close_price = self.close.shift(-3).values
            self.scaler = preprocessing.MinMaxScaler()
            self.validation_df, self.train_df = self.preprocess(self.validation_df, self.scaler), self.preprocess(self.train_df, self.scaler)
            
            # # Building Training Data
            train_seq = self.build_sequences(self.train_df)
            X_train, y_train = self.extract_feature_labels(train_seq)

            # Building Validation Data
            validate_seq = self.build_sequences(self.validation_df, val=True)
            X_test, y_test = self.extract_feature_labels(validate_seq)

            self.df = pd.DataFrame(data=self.validation_df)

            self.y_arr =np.zeros((len(y_test), len(self.validation_df[0])))
            self.build_model(X_train, y_train, X_test, y_test) 

        else:
            print("Dataset {} not found. Program exiting".format(filename))

    def preprocess(self, df, scaler):
        # s = preprocessing.MinMaxScaler().fit(df.values)
        df_minmax = scaler.fit_transform(df.values)
        return df_minmax

    def get_percentage_change(self, values):
        return (values-values.shift(1))/values.shift(1) 

    def get_fivepct(self):
        self.dataset =self.dataset.sort_index()
        # self.dataset = self.dataset[::-1]
        last_5pct = int(len(self.dataset.index)*(0.05))
        index =self.dataset.iloc[-last_5pct].name 
        train_df = self.dataset.truncate(after=index)
        validation_df= self.dataset.truncate(before=index)
        return validation_df, train_df

    def build_sequences(self, df, val= False):
        sequential_data = []
        prev_days = deque(maxlen = PriceClassification.SEQ_LENGTH)
        for i in df:
            prev_days.append([n for n in i[:-1]])
            if len(prev_days) == PriceClassification.SEQ_LENGTH:
                sequential_data.append([np.array(prev_days), i[-1]])
        if val:
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
        print("TRAIN: Amount of Sells: {}, Amount of Buys: {}, Amount of holds: na".format(y_train.count(0), y_train.count(1)))
        print("VALIDATE: Amount of Sells: {}, Amount of Buys: {}, Amount of holds: na".format(y_val.count(0), y_val.count(1)))


    def build_model(self, X_train, y_train, X_test, y_test):
        print(f'X_train {X_train.shape[1:]}')
        model = Sequential()

        model.add(CuDNNLSTM(32, input_shape=(X_train.shape[1:]), return_sequences=True))
        model.add(Dropout(0.25))
        model.add(CuDNNLSTM(32))
        model.add(Dense(1))
        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        model.compile(loss='mse',optimizer=opt )

        tensorboard = TensorBoard(log_dir='logs/{}'.format(PriceClassification.NAME))
        filepath = "RNN_Final-{epoch:02d}-{loss:.3f}"
        checkpoint = ModelCheckpoint("../models/{}.model".format(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')) 

        model.fit(X_train, y_train, epochs=50, batch_size=16, shuffle=False, callbacks=[tensorboard, checkpoint])

        predicted_price = model.predict(X_test)
        predicted_list = []
        for i in range(len(predicted_price)):
            predicted_list.append(predicted_price[i][0])

        self.y_arr[:, 13] = predicted_list
        predicted_price = self.scaler.inverse_transform(self.y_arr)
        print(predicted_price[:, 13])
        t = np.arange(0.0, len(predicted_price[:, 13]))

        post_df = pd.DataFrame(index= self.close_price.index)
        post_df['actual +3 Hour'] =self.close_price.values
        post_df['predicted +3 Hour'] = predicted_price[:, 13]
        post_df.to_csv('post.csv')


        # model.add(Activation('linear'))

        # opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        # model.compile(loss='mae', optimizer=opt, metrics=['accuracy'])

        # model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())

        # model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
        # model.add(Dropout(0.1))
        # model.add(BatchNormalization())

        # model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:])))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())


        


        # filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
        # checkpoint = ModelCheckpoint("../models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) 
        # model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=1, verbose=2, shuffle=True, callbacks=[tensorboard, checkpoint])

        # # history = model.fit(X_train, y_train,
        # #                 batch_size=PriceClassification.BATCH_SIZE,
        # #                 epochs=PriceClassification.EPOCHS,
        # #                 validation_data=(X_val, y_val),
        # #                 callbacks=[tensorboard, checkpoint])

        # # Score model
        # score = model.evaluate(X_val, y_val, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
        # # Save model
        # model.save("../models/{}".format(PriceClassification.NAME))
