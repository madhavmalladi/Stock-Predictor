from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from predict import predict_stocks, get_model_path

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

MODEL_PATH = get_model_path()
if MODEL_PATH:
    print(f"Using model: {MODEL_PATH}")
else:
    print("No model found. You may need to train the model first.")

@app.route('/api/predict', methods=['OPTIONS'])
def handle_options():
    return '', 200

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

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
        
        if not ticker:
            return jsonify({
                "error": "No ticker provided",
                "status": "error"
            }), 400
        
        # Check if model exists
        if not MODEL_PATH:
            return jsonify({
                "error": "No trained model found. Please train the model first.",
                "status": "error"
            }), 500
        
        # Make prediction
        result = predict_stocks([ticker], MODEL_PATH)
        
        if ticker not in result:
            return jsonify({
                "error": f"Failed to predict for {ticker}",
                "status": "error"
            }), 400
            
        if "error" in result[ticker]:
            return jsonify({
                "error": result[ticker]["error"],
                "status": "error"
            }), 400
        
        prediction = result[ticker]
        
        return jsonify({
            "ticker": ticker,
            "current_price": prediction["current_price"],
            "predicted_price": prediction["predicted_price"],
            "change_percentage": prediction["change_percentage"],
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/predict-multiple', methods=['POST'])
def predict_multiple():
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({
                "error": "No tickers provided",
                "status": "error"
            }), 400
        
        # Check if model exists
        if not MODEL_PATH:
            return jsonify({
                "error": "No trained model found. Please train the model first.",
                "status": "error"
            }), 500
        
        # Make predictions
        results = predict_stocks(tickers, MODEL_PATH)
        
        # Format response
        formatted_results = []
        for ticker, prediction in results.items():
            if "error" in prediction:
                formatted_results.append({
                    "ticker": ticker,
                    "error": prediction["error"],
                    "status": "error"
                })
            else:
                formatted_results.append({
                    "ticker": ticker,
                    "current_price": prediction["current_price"],
                    "predicted_price": prediction["predicted_price"],
                    "change_percentage": prediction["change_percentage"],
                    "status": "success"
                })
        
        return jsonify({
            "predictions": formatted_results,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
    
if (__name__ == '__main__'):
    app.run(debug = True, port = 8000)