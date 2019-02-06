import run_model
import pandas as pd
import json
import time 

def get_latest_TA_data(coin):
    df = pd.read_csv(f'data_files/visual/{coin}_raw.csv') 
    result = {
        'coin' : coin,
        'date' : df.date.iloc[-1],
        'indicators' : df.iloc[-1].to_dict()
    }

    with open('data/coin_data.json', 'w') as f:
        json.dump(result, f)
    return result
