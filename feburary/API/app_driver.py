import run_model
import pandas as pd
import json
import time


def get_latest_TA_data(coin):
    result_list = []
    df = pd.read_csv(f'data_files/visual/{coin}_raw.csv') 

    print('Updating data....')
    time_diff = int(time.time()) - df.date.iloc[-1]
    if time_diff > 60:
        run_model.run('realtime', coin)
        df = pd.read_csv(f'data_files/visual/{coin}_raw.csv') 

    df = pd.read_csv(f'data_files/visual/{coin}_raw.csv') 
    result = {
        'coin' : coin,
        'date' : df.date.iloc[-1],
        'signal': 'buy',
        'indicators' : df.iloc[-1].to_dict()
    }
    result_list.append(result)

    with open('data/display_data.json', 'w') as f:
        json.dump(result_list, f)
    return result


def get_latest_signal(coin):
    df = pd.read_csv(f'data_files/visual/{coin}_raw.csv') 
    time_diff = int(time.time()) - df.date.iloc[-1]
    if time_diff > 60:
        print('Need to update')
    run_model.run_signals(coin)

# get_latest_signal("BTC")