import asyncio
import datetime
import random
import websockets
import time 
import run_model
import pandas as pd
import json
import _thread

async def ws_signal_data(websocket, path):
    increment = 0
    coin = 'BTC'
    while True:
        df = pd.read_csv(f'data_files/visual/{coin}_signals.csv', index_col=0) 
        time_diff = int(time.time()) - df.index[-1]
        if time_diff > 60:
            print('Need to update')
            run_model.run_signals(coin)

        df_s = pd.read_csv(f'data_files/visual/{coin}_signals.csv') 
        obj = df_s.values
        result_dict = {}

        data = {
            'type':'signal', 
            'message' : obj.tolist()
        }
        send_loaded = json.dumps(data)
        await websocket.send(send_loaded)
        await asyncio.sleep(10)

def change_buy(list_):
    list__ = []
    for i in list_:
        if i == 1:
            list__.append('Buy')
        elif i == 0:
            list__.append('Sell')
        else:
            list__.append(i)
    return list__
start_server = websockets.serve(ws_signal_data, '127.0.0.1', 5679)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()