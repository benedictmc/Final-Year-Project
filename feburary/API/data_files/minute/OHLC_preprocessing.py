import pandas as pd 
import numpy as np
import plotly as py
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from datetime import datetime
import scipy.optimize
import warnings
import matplotlib.pyplot as plt
import math
import talib
from scipy import stats
import json

class OHLCPreprocess():
    def __init__(self):
        self.METHOD_CALLS = {
            'moving_averages' : self.moving_averages,
            "rsi": self.rsi,
            'stochastic': self.stochastic,
            'macd' : self.macd,
            'williams': self.williams,
            "bollinger_bands" : self.bollinger_bands,
        }
        ##Initialse dataframe from master BTC datafile. Datafile is in minute time periods
        read_filename = 'data_files/hour/master_dataset_BTC.csv'
        print('Running OHLC preprocessing ...')
        self.data = pd.read_csv(read_filename, index_col=0)

        ##Reduces row size to 200 for time sake        
        self.data.index = [datetime.fromtimestamp(int(x)).strftime('%d.%m.%Y %H:%M:%S') for x in self.data.index]
        ##Creates an index variable as this will be used often
        self.index =  self.data.index
        self.all_data = self.combine_indicators() 

        change_df = pd.DataFrame(index=self.index)
        diff_list = ['ma_5','ma_6','ma_10','%b_bands','midlle_bands','rsi_6','rsi_12','williams_12','9_kstochastic','9_dstochastic','macd_hist']

        for col in diff_list:
            if col != 'close':
                change_df[f'%_{col}'] = self.get_percentage_change(self.all_data[col])

        self.all_data = pd.concat([self.all_data, change_df], axis=1, join='inner')
        self.all_data = self.all_data.replace([np.inf, -np.inf], np.nan)
        self.all_data.dropna(inplace=True)

        # self.build_classification('BTC')
        # length = len(self.all_data)
        # min_max_arr = self.preprocess(self.all_data)
        # for i in range(length):
        #     np.append(min_max_arr[i], int(self.all_data.target.iloc[i]))

        # self.post_process_df = pd.DataFrame(min_max_arr)
        # self.post_process_df.to_csv(filename)

    def combine_indicators(self):
        all_df = pd.DataFrame(index=self.data.index)
        all_df['close'] = self.data.Close

        with open('metadata/period_lists.json', 'r') as f:
            data_list = json.load(f)

        for method in self.METHOD_CALLS:
            if method == 'acc_dist': continue
            period_list = data_list[method]
            function = self.METHOD_CALLS[method]
            if len(period_list) != 0:
                for period in period_list:
                    print('Calling method {} for a period of {}'.format(method, str(period)))
                    df = function(period)
                    if all_df.empty:
                        all_df = df
                    else:
                        all_df = pd.concat([all_df, df], axis=1, join='inner')
            else: 
                print('Calling method {} for a period of {}'.format(method, str(period)))
                if all_df.empty:
                    all_df = function()
                else:
                    all_df = pd.concat([all_df, function()], axis=1, join='inner')
        all_df.dropna(inplace=True)
        return all_df


    def moving_averages(self, period):
        df = pd.DataFrame(index=self.index)
        close = self.data.Close    
        df[f'ma_{period}'] = close.rolling(period).mean()
        return df       
        
    def rsi(self, period):
        df = pd.DataFrame(index=self.index)
        close = self.data.Close    
        rsi = talib.RSI(close, timeperiod=14)    
        df[f'rsi_{period}'] = rsi
        return df

    ##Calculates heikanashi candles from OHLC data 
    def heikanashi_candles(self):
        ##Heikanashi close is the sum of OHLC/4
        HAclose = self.data[['Open', 'High', 'Low', 'Close']].sum(axis=1)/4
        HAopen = HAclose.copy()
        HAopen.iloc[0] = HAclose.iloc[0]
        HAlow = HAclose.copy()
        HAhigh = HAclose.copy()

        for i in range(1, len(self.data.index)):
            ##Heikanashi open open minus close  / 2 
            HAopen.iloc[i] = (HAopen.iloc[i-1]+HAclose.iloc[i-1])/2
            ##Heikanashi high is the max of the high, HAopen and HAclose
            HAhigh.iloc[i] = np.array([self.data.High.iloc[i], HAopen.iloc[i],  HAclose.iloc[i]]).max()
            ##Heikanashi low is the min of the low, HAopen and HAclose
            HAlow.iloc[i] = np.array([self.data.Low.iloc[i], HAopen.iloc[i],  HAclose.iloc[i]]).min()
        
        ##Creates new dataframe with heininashi candles
        df = pd.concat((HAopen, HAhigh, HAlow, HAclose), axis = 1)
        df.columns = [['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]

        return df

    ##Default method is diffence, any other method passed in will do a linear regression approach to detrend data
    def detrend_data(self, method='difference'):
        prices = self.data.Close
        if method == 'difference':
            print('Using difference method')
            ##Subracts previous price from the current price 
            detrended = prices - prices.shift(1)
        else:
            print('Using linear method')
            x = np.arange(0,len(prices)).reshape(-1,1)
            y = prices.values.reshape(-1,1)
            model = LinearRegression()
            model.fit(x,y)
            trend = model.predict(x)
            trend = trend.reshape((len(prices),))
            detrended = prices - trend
        return detrended

    ##Fourier fit method maybe could be a lamda function instead
    def fourier_fit(self, x, a0, a1, b1, w): return a0 + a1*np.cos(w*x) + b1*np.sin(w*x)

    ##Sin series function maybe could be a lamda function instead
    def sin_series(self, x, a0, b1, w): return a0 + b1*np.sin(w*x)
    
    ##Fourier fit function
    def fourier_function(self):
        ## Periods is the list of periods to compute the fourier function on 
        periods = [10, 15]
        prices = self.data.Close

        fourier_dict = {}
        detrended = self.detrend_data()
        plot = False
        print('Starting fourier fucntion ...')
        for period in periods:
            coeffs = []
            for j in range(period, len(detrended)-period):
                x = np.arange(0, period)
                y = detrended.iloc[j - period:j]
                y.fillna(0, inplace=True)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', scipy.optimize.OptimizeWarning)
                    try:
                        result = scipy.optimize.curve_fit(self.fourier_fit,x,y.values)
                    except:
                        result = np.empty((1,4))
                        result[0,:] =np.NAN

                if plot:
                    xt = np.linspace(0, period, 100)
                    yt = self.fourier_fit(xt, result[0][0], result[0][1], result[0][2], result[0][3])

                    plt.plot(x,y)
                    plt.plot(xt,yt,'r')
                    plt.show()

                coeffs = np.append(coeffs, result[0], axis=0)

            coeffs = np.array(coeffs).reshape(len(coeffs)/4,4)
            df = pd.DataFrame(coeffs, index =self.index)
            df.columns = [['a0','a1','b1','w']]
            df.fillna(method='bfill', inplace=True)
            fourier_dict[period] = df

    def sin_function(self):
        ## Periods is the list of periods to compute the fourier function on 
        periods = [10, 15]
        prices = self.data.Close
        fourier_dict = {}
        detrended = self.detrend_data()
        plot = True
        print('Starting sin fucntion ...')
        for period in periods:
            coeffs = []
            for j in range(period, len(detrended)-period):
                x = np.arange(0, period)
                y =  detrended.iloc[j - period:j]
                y.fillna(0, inplace=True)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', scipy.optimize.OptimizeWarning)
                    try:
                        result = scipy.optimize.curve_fit(self.sin_series,x,y)
                        print('Sucess paramameters saved')
                    except:
                        result = np.empty((1,3))
                        result[0,:] =np.NAN

                if plot:
                    xt = np.linspace(0, period, 100)
                    yt = self.sin_series(xt,result[0][0], result[0][1], result[0][2])

                    plt.plot(x,y)
                    plt.plot(xt,yt,'r')
                    plt.show()

                coeffs = np.append(coeffs, result[0], axis=0)

            coeffs = np.array(coeffs).reshape(len(coeffs)/3, 3)
            df = pd.DataFrame(coeffs, index =prices.iloc[period:-period])
            df.columns = [['a0','b1','w']]
            df.fillna(method='bfill', inplace=True)
            fourier_dict[period] = df

    def williams_dist(self):
        high= self.data.High
        low = self.data.Low
        close = self.data.Close
        vol = self.data.Volume
        WAD = []
        WAD.append(0)
        for i in range(1,len(high)):
            cc, lc, c_vol = close.iloc[i] , close.iloc[i-1], vol.iloc[i]
            TRhigh = np.array([high.iloc[i], lc]).max()
            TRlow = np.array([low.iloc[i], lc]).min()
            if cc > lc:
                price_move = cc - TRlow
            elif cc < lc:
                price_move = cc - TRhigh
            elif cc == lc:
                price_move = 0
            else:
                print('Error encountered getting price move')
        
            AD = price_move*c_vol
            WAD = np.append(WAD, AD)

        WAD = WAD.cumsum()
        wad_df = pd.DataFrame(WAD, index=self.data.index)
        wad_df.columns = [['wad']]

        print(wad_df)

    def stochastic(self, period):
        df = pd.DataFrame(index=self.index)
        high, low, close = self.data.High, self.data.Low, self.data.Close
        slowk, slowd = talib.STOCH(high, 
                            low, 
                            close, 
                            fastk_period=period, 
                            slowk_period=3, 
                            slowk_matype=0, 
                            slowd_period=3, 
                            slowd_matype=0)
        if period == 9:
            df['9_kstochastic'] = slowk
            df['9_dstochastic'] = slowd
        else:
            ##Returning the difference between K line and D line. This means closer to zero means a crossover
            df['stochastic_{}'.format(str(period))] = (slowk - slowd)
        
        return df

    def momentum(self, period):
        df = pd.DataFrame(index=self.index)
        close = self.data.Close
        momentum = talib.MOM(close, timeperiod=period)
        df['momentum_{}'.format(str(period))] = momentum
        return df
            

    def williams(self, period):
        df = pd.DataFrame(index=self.index)
        high, low, close = self.data.High, self.data.Low, self.data.Close
        williams = talib.WILLR(high, low, close, timeperiod=period)
        df['williams_{}'.format(str(period))] = williams
        return df

    def PROC(self):
        df = pd.DataFrame(index=self.index)
        close = self.data.Close
        percent_change = (close - close.shift(1))/close.shift(1)
        df['%_change'] = percent_change
        return df

    def get_percentage_change(self, values):
        return (values-values.shift(1))/values.shift(1) 
   
    ##TODO need to fix so current index is not being cut off
    def acc_dist(self, period):
        AD = []
        high, low, close, volume = self.data.High, self.data.Low, self.data.Close, self.data.Volume
        for i in range(period, len(self.data.index)-period):
            C = close.iloc[i+1]
            H = np.array([high.iloc[i-period: period+i]]).max()
            L = np.array([low.iloc[i-period: period+i]]).min()
            V = volume.iloc[i+1]
            if H==L:
                CLV = 0
            else: 
                CLV = (((C-L)-(H-C))/(H-L))
            AD = np.append(AD, CLV*V)
        AD = AD.cumsum()
        AD = pd.DataFrame(AD, index=self.data.iloc[period+1:-period+1].index)
        return AD

    def macd(self):
        df = pd.DataFrame(index=self.index)
        close = self.data.Close
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=15, slowperiod=30, signalperiod=9)
        df['macd_hist'] = macdhist
        return df


    def cci(self):
        df = pd.DataFrame(index=self.index)
        high, low, close = self.data.High, self.data.Low, self.data.Close
        cci = talib.CCI(high, low, close, timeperiod=14)
        df['cci'] = cci
        return df

    def bollinger_bands(self):
        df = pd.DataFrame(index=self.index)
        close = self.data.Close
        upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=15, nbdevup=2, nbdevdn=2, matype=0)
        df['%b_bands'], df['midlle_bands'] = ((close-lowerband)/(upperband-lowerband)), middleband
        return df


    def averages(self,period):
        average_df= pd.DataFrame(self.data[['Open', 'High', 'Low', 'Close']].rolling(period).mean(), index=self.index)
        average_df.columns = ['O_avg_{}'.format(period), 'H_avg_{}'.format(period), 'L_avg_{}'.format(period), 'C_avg_{}'.format(period)]
        return average_df
    

    def slopes(self, period):
        df = pd.DataFrame(index=self.index)
        high, ms = self.data.High, []
        for i in range(period, len(self.data.index) - period):
            y = high.iloc[i - period:period+i].values
            x = np.arange(0, len(y))
            res = stats.linregress(x, y=y)
            m = res.slope
            ms = np.append(ms, m )

        ms_df = pd.DataFrame(ms, index= self.data.iloc[period:-period].index)
        return ms_df

    def build_classification(self, coin):
        print("Getting target classification information")
        self.all_data['future'] = self.all_data.close.shift(-3)
        self.all_data['target'] = list(map(self.get_class, self.all_data.close,  self.all_data['future']))
        self.all_data = self.all_data.drop('future', 1)

    def get_class(self, current, future):
        if float(current)< float(future) :
            return 1
        else:
            return 0

    def preprocess(self, df):
        minmax_scale = preprocessing.MinMaxScaler().fit(df)
        df_minmax = minmax_scale.transform(df)
        return df_minmax


x = OHLCPreprocess()