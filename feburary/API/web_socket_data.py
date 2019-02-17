import asyncio
import datetime
import random
import websockets
import time 
import run_model
import pandas as pd
import json
import _thread

async def ws_TA_data(websocket, path):
    coin = 'BTC'
    while True:
        df = pd.read_csv(f'data_files/visual/{coin}_raw.csv') 
        time_diff = int(time.time()) - df.date.iloc[-1]
        if time_diff > 60:
            run_model.run('realtime', coin)

        result_list = []
        df = pd.read_csv(f'data_files/visual/{coin}_raw.csv') 
        result = {
            'coin' : coin,
            'date' : df.date.iloc[-1],
            'signal': 'buy',
            'indicators' : df.iloc[-1].to_dict()
        }
        result_list.append(result)
        data = {
            'type':'display data',
            'message' : result_list
        }

        send_loaded = json.dumps(data)
        await websocket.send(send_loaded)
        await asyncio.sleep(10)


start_server = websockets.serve(ws_TA_data, '127.0.0.1', 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()