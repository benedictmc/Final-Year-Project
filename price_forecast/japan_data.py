import pandas as pd

class JapanData():
    def __init__(self):
        df = pd.read_csv('data_files/training/japan_btc.csv')
        df.index = pd.date_range('1/1/2018', periods=len(df), freq='S')
        df = df.resample('10S').ohlc()
        df = df.btc
        print(df.head())
x = JapanData()