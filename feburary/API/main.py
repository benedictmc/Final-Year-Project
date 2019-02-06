from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import jwt
from datetime import datetime  
import app_driver

app = Flask(__name__)
CORS(app)

with open('data/metadata.json', 'r') as f:
    metadata = json.load(f)


@app.route('/API/coins')
def get_coins():
    return jsonify(metadata['coins'])

@app.route('/API/features')
def get_fetaures():
    return jsonify(metadata['features'])

@app.route('/API/feature-data')
def get_fetaure_data():
    return jsonify(app_driver.get_latest_TA_data('BTC'))



if __name__ == "__main__":
        app.run()