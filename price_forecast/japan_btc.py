import pandas as pd 

class JapanBTC():
    def __init__(self):
        df = pd.read_csv('data_files/post/japan.csv', index_col=0)
        df = self.post_df(df)
        df.to_csv('data_files/post/japan_post.csv')
        self.run_profit_loss(df)

    def resample_data(self):
        df = pd.read_csv('data_files/training/japan_btc.csv')
        df.index = pd.date_range('1/1/2018', periods=len(df), freq='S')
        df = df.resample('10S').ohlc()
        df = df.btc
        df.to_csv('data_files/training/japan_btc_post.csv')

    def post_df(self, df):
        df.index = pd.DatetimeIndex(df.index)
        idx = pd.date_range(df.index[0], df.index[-1], freq='10S')
        df = df.reindex(idx, fill_value=0)
        df['compare'] = df.actual.shift(3)
        df['buy_sell'] = [1 if row[1].predicted > row[1].actual else 0 for row in df.iterrows()]
        print(df.buy_sell.value_counts(0))
        print(df)

        return df

    def run_profit_loss(self, df):
        bal, bought, sold = 6300, False, True
        for row in df.iterrows():
            if row[1].buy_sell == 1 and sold:
                if row[1].actual == 0:
                    continue
                bal = bal * row[1].actual
                bought, sold = True, False
                print(f'BUYING: {bal} at price {row[1].actual}')
            if row[1].buy_sell == 0  and bought:
                if row[1].actual == 0:
                    continue
                bal = bal / row[1].actual
                bought, sold = False, True
                print(f'SELLING: {bal} at price {row[1].actual}')
        print(f'Final Balance is {bal}')

x = JapanBTC()