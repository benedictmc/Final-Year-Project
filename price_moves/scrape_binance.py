

from binance.client import Client
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import talib
from pytrends.request import TrendReq

class ScrapeBinance():
    API, API_SECRET = '', ''

    def __init__(self):
        self.link = 'https://www.binance.com/en/trade/RVN_BTC'

        with open('../metadata/binance_keys.txt') as f:
            keys = f.read()
            keys = keys.split(',')
            ScrapeBinance.API, ScrapeBinance.API_SECRET = keys[0], keys[1]
        self.client = Client(ScrapeBinance.API, ScrapeBinance.API_SECRET) 
        self.fill_json()


    def fill_json(self):
        ticker_list = [symbol['symbol'] for symbol in self.client.get_all_tickers()]
        ticker_list = set([ticker[:-3] for ticker in ticker_list if ticker[-4:] != 'USDT'])
        with open('meta/coin.json','r') as f:
            coin_info = json.load(f)
        coin_list = list(coin_info.keys())

        for new in list(ticker_list.difference(set(coin_info))):
            coin_info[new] = ''
        print(coin_info)
        with open('meta/coin.json','w') as f:
            json.dump(coin_info, f)



x = ScrapeBinance()