import pandas as pd
import numpy as np
from collections import deque


class TestResults():
    def __init__(self):
        filename = 'data_files/training/BTC_minute.csv'
        self.dataset = pd.read_csv(filename, index_col=0)
        self.start_method()

        self.validation_df, _ = self.get_fivepct()
        # self.close_price = self.validation_df.iloc[4:].close
        # self.close_price = self.close.shift(-3).values
        print(self.validation_df.columns)
        validate_seq = self.build_sequences(self.validation_df)
        X_test, y_test = self.extract_feature_labels(validate_seq)

        # self.y_arr =np.zeros((len(y_test), len(self.validation_df[0])))

    def get_fivepct(self):
        self.dataset =self.dataset.sort_index()
        last_5pct = int(len(self.dataset.index)*(0.05))
        index =self.dataset.iloc[-last_5pct].name 
        train_df = self.dataset.truncate(after=index)
        validation_df= self.dataset.truncate(before=index)
        return validation_df, train_df

    def preprocess(self, df, scaler):
        # s = preprocessing.MinMaxScaler().fit(df.values)
        df_minmax = scaler.fit_transform(df.values)
        return df_minmax

    def start_method(self):
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

    def get_percentage_change(self, values):
        return (values-values.shift(1))/values.shift(1) 


    def build_sequences(self, df):
        sequential_data = []
        prev_days = deque(maxlen = 5)
        for i in df:
            print(i)
            prev_days.append([n for n in i[:-1]])
            if len(prev_days) == 5:
                sequential_data.append([np.array(prev_days), i[-1]])
        return sequential_data


    def extract_feature_labels(self, seq_data):
        X, y = [], []
        for seq, target in seq_data:
            X.append(seq)
            y.append(target)
        return np.array(X), y
x = TestResults()