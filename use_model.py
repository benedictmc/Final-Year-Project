import os
from tensorflow.keras.models import Sequential, load_model
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque

class UseModel():
    SEQ_LENGTH = 60
    def __init__(self):
        dataset_filepath = "dataset_files/dataset_20-15-02_XRP_min.csv"
        filepath = "models\RNN_Final-15-0.548.model"
        self.dataset = pd.read_csv(dataset_filepath, index_col=0)
        self.dataset = self.dataset.drop('future', 1)
        self.test_df, _ = self.get_fivepct()
        self.test_df = self.preprocess(self.test_df)
        test_seq = self.build_sequences(self.test_df)
        X, y = self.extract_feature_labels(test_seq)
        model = load_model(filepath)

        predicted_y = model.predict_classes(X)
        true_count, false_count = 0, 0
        for i in range(len(predicted_y)):
            if predicted_y[i] == y[i]:
                true_count+=1
            else:
                false_count+=1
        print("There was {} good results, and {} bad results".format(true_count, false_count))
            # print("The predicted ouput was {} and the real output was {}".format(predicted_y[i], y[i]))


    def get_fivepct(self):
        times = sorted(self.dataset.index.values)
        last_5pct = times[-int(len(times)*(0.1))]
        test_df = self.dataset[(self.dataset.index >= last_5pct)]
        train_df = self.dataset[(self.dataset.index < last_5pct)]
        return test_df, train_df


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
        prev_days = deque(maxlen = UseModel.SEQ_LENGTH)
        for i in df.values:
            prev_days.append([n for n in i[:-1]])
            if len(prev_days) == UseModel.SEQ_LENGTH:
                sequential_data.append([np.array(prev_days), i[-1]])
        return sequential_data

    
    def extract_feature_labels(self, seq_data):
        X, y = [], []
        for seq, target in seq_data:
            X.append(seq)
            y.append(target)
        return np.array(X), y

x = UseModel()