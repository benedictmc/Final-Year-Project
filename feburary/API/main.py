from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
from datetime import datetime  
import app_driver
import json

app = Flask(__name__)
CORS(app)



with open('data/metadata.json', 'r') as f:
        metadata = json.load(f)


@app.route('/API/coins')
def get_coins():
    return jsonify(metadata['coins'])

@app.route('/API/features')
def get_fetaures():
    print(metadata['features'])
    return jsonify(metadata['features'])

@app.route('/API/feature-data')
def get_fetaure_data():
    return jsonify(app_driver.get_latest_TA_data('BTC'))


@app.route('/API/test')
def get_test():
    with open('data/TEST.json', 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/API/display-data')
def get_display_data():
    app_driver.get_latest_TA_data('BTC')
    with open('data/display_data.json', 'r') as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/API/signal-data')
def get_signal_data():
    app_driver.get_latest_signal('BTC')
    with open('data/signal_data.json', 'r') as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == "__main__":
        app.run()