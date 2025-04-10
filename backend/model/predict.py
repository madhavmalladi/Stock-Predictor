import os
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from model import StockPredictor

def get_model_path():
    try:
        # Absolute path to model's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        model_dir = os.path.join(parent_dir, 'saved_models')
        
        if not os.path.exists(model_dir):
            print(f"Model directory not found at {model_dir}")
            return None
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        if not model_files:
            print(f"No model files found in {model_dir}")
            return None
        
        if 'final_model.h5' in model_files:
            return os.path.join(model_dir, 'final_model.h5')
        
        checkpoint_files = [f for f in model_files if f.startswith('model_epoch_')]
        if not checkpoint_files:
            return None
        
        latest_checkpoint = sorted(checkpoint_files, 
                                 key=lambda x: int(x.split('_')[2].split('.')[0]), 
                                 reverse=True)[0]
        return os.path.join(model_dir, latest_checkpoint)
    
    except Exception as e:
        print(f"Error finding model: {str(e)}")
        return None

def load_model(model_path=None):
    """Load the trained stock prediction model"""
    if model_path is None:
        model_path = get_model_path()
        
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    try:
        print(f"Loading model from {model_path}")
        # Create a new model with the same architecture and load weights directly
        features = ['Close', 'Volume', 'SMA_7', 'SMA_20', 'Daily_Return', 'Volatility', 'Volume_MA']
        sequence_length = 20
        
        model = StockPredictor(input_shape=(sequence_length, len(features)))
        
        try:
            model.model.load_weights(model_path)
            print("Loaded model weights successfully")
        except:
            # try with compile=false
            loaded_model = tf.keras.models.load_model(model_path, compile=False)
            model.model.set_weights(loaded_model.get_weights())
            
        return model.model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def calculate_technical_indicators(df):
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Volume Moving Average
    df['Volume_MA'] = df['Volume'].rolling(window=7).mean()
    
    return df

def predict_stock(ticker, model=None, period='6mo'):
    """Predict future stock price for a given ticker"""
    try:
        if model is None:
            model = load_model()
            
        features = ['Close', 'Volume', 'SMA_7', 'SMA_20', 'Daily_Return', 'Volatility', 'Volume_MA']
        sequence_length = 20
        
        # Avoiding rate limiting
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Download the data
        df = yf.download(ticker, period=period, progress=False, session=session)
        
        if df.empty or len(df) < sequence_length + 1:
            raise ValueError(f"Not enough data available for {ticker}")
            
        df = calculate_technical_indicators(df)
        current_price = df['Close'].iloc[-1]
        data = df[features].dropna()
        
        if len(data) < sequence_length + 1:
            raise ValueError(f"Not enough processed data available for {ticker}")
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        input_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, len(features))
        predicted_scaled = model.predict(input_sequence)[0][0]
        
        last_normalized_close = scaled_data[-1, 0]
        
        pct_change = (predicted_scaled - last_normalized_close) / last_normalized_close
        
        predicted_price = current_price * (1 + pct_change)
        
        return {
            'ticker': ticker,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'change_percentage': float(pct_change * 100)
        }
        
    except Exception as e:
        print(f"Error predicting price for {ticker}: {str(e)}")
        raise e

def predict_stocks(tickers, model_path=None):
    # Multiple tickers
    try:
        model = load_model(model_path)
        results = {}
        
        for ticker in tickers:
            try:
                prediction = predict_stock(ticker, model)
                results[ticker] = prediction
                print(f"Successfully predicted for {ticker}")
            except Exception as e:
                print(f"Error predicting for {ticker}: {str(e)}")
                results[ticker] = {"error": str(e)}
                
        return results
    except Exception as e:
        raise Exception(f"Error predicting stocks: {str(e)}")

if __name__ == "__main__":
    # Test prediction functionality
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NFLX', 'AMZN']
    
    try:
        results = predict_stocks(test_tickers)
        
        print("\nPrediction Results:")
        for ticker, prediction in results.items():
            if "error" in prediction:
                print(f"{ticker}: Error - {prediction['error']}")
            else:
                current = prediction.get('current_price', 'N/A')
                predicted = prediction.get('predicted_price', 'N/A')
                change = prediction.get('change_percentage', 'N/A')
                
                print(f"{ticker}: Current: ${current:.2f}, Predicted: ${predicted:.2f}, Change: {change:.2f}%")
    except Exception as e:
        print(f"Error during testing: {str(e)}")
