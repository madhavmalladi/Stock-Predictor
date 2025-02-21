from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/', methods = ['GET'])
def home():
    return jsonify({
        "message": "Welcome to the Stock Predictor",
        "status": "success"
    })

@app.route('/api/test', methods = ['GET'])
def test():
    return jsonify({
        "message": "Backend works",
        "status": "success"
    })

@app.route('/api/predict', methods = ['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        return jsonify({
            "ticker": ticker,
            "prediction": [100.0, 101.0, 102.0],
            "message": "prediction successful"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
    
if (__name__ == '__main__'):
    app.run(debug = True, port = 5000)