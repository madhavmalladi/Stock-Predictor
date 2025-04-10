import os
import pandas as pd
import numpy as np
import yfinance as yf
import time
import random
import requests
from model import StockPredictor

def get_tickers(numStocks = 5):
    popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    print(f"Using {len(popular_tickers[:numStocks])} popular tickers")
    return popular_tickers[:numStocks]

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
    
def train_model(numStocks = 5, epochs = 10, batch_size = 32, save_path = 'saved_models'):
    tickers = get_tickers(numStocks)
    features = ['Close', 'Volume', 'SMA_7', 'SMA_20', 'Daily_Return', 'Volatility', 'Volume_MA']
    sequence_length = 20
    
    model = StockPredictor(input_shape=(sequence_length, len(features)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for epoch in range(epochs):
        totalLoss = 0
        totalMae = 0
        successfulTickers = 0

        for ticker in tickers:
            try:
                print(f"\nTraining on {ticker} - Epoch {epoch+1}/{epochs}")
                
                # Avoiding rate limiting
                time.sleep(random.uniform(3.0, 5.0))
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                
                df = yf.download(ticker, period='1y', progress=False, session=session)
                
                if df.empty or len(df) < batch_size + 20:  # Need enough data for sequences
                    print(f"Not enough data for {ticker}. Skipped")
                    continue
                    
                df = calculate_technical_indicators(df)
                data = df[features].dropna()
                if len(data) < batch_size + 20:
                    print(f"Not enough processed data for {ticker}. Skipped")
                    continue
                
                last_price = df['Close'].iloc[-1]
                
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)
                
                X = []
                y = []
                for i in range(len(scaled_data) - sequence_length):
                    X.append(scaled_data[i:(i + sequence_length)])
                    y.append(scaled_data[i + sequence_length, 0])  # 0 is Close price
                
                X = np.array(X)
                y = np.array(y)
                
                if len(X) < batch_size:
                    print(f"Not enough sequence data for {ticker}")
                    continue
                    
                history = model.fit(X, y, epochs=1, batch_size=batch_size, validation_split=0.2, verbose=1)
                totalLoss += history.history['loss'][0]
                totalMae += history.history['mae'][0]
                successfulTickers += 1
                
                print(f"Successfully trained on {ticker}")
                
            except Exception as e:
                print(f"Error: {str(e)} for ticker {ticker}")
                continue

        if successfulTickers > 0:
            avgLoss = totalLoss / successfulTickers
            avgMae = totalMae / successfulTickers
            print(f'\nEpoch [{epoch+1}/{epochs}]')
            print(f'Stocks processed successfully: {successfulTickers}/{len(tickers)}')
            print(f'Average Loss: {avgLoss:.4f}')
            print(f'Average MAE: {avgMae:.4f}')
        
        if (epoch+1) % 5 == 0:  # Save checkpoints more frequently
            checkpointPath = os.path.join(save_path, f'model_epoch_{epoch+1}.h5')
            model.save(checkpointPath)

    finalModelPath = os.path.join(save_path, 'final_model.h5')
    model.save(finalModelPath)

    return model
    
def evaluate_model(model, n_test_stocks = 2):
    testTickers = ['NFLX', 'DIS'][:n_test_stocks]
    
    features = ['Close', 'Volume', 'SMA_7', 'SMA_20', 'Daily_Return', 'Volatility', 'Volume_MA']
    sequence_length = 20
    
    results = {}
    for ticker in testTickers:
        try:
            time.sleep(random.uniform(3.0, 5.0))
            
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            df = yf.download(ticker, period='6mo', progress=False, session=session)
            if df.empty:
                print(f"No data for {ticker}")
                continue
                
            df = calculate_technical_indicators(df)
            data = df[features].dropna()
            if len(data) < 30:
                print(f"Not enough processed data for {ticker}")
                continue
            
            last_price = df['Close'].iloc[-1]
            
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            X = []
            y = []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:(i + sequence_length)])
                y.append(scaled_data[i + sequence_length, 0])  # 0 is Close price
            
            X = np.array(X)
            y = np.array(y)
            
            predictions = model.predict(X)
            
            mse = np.mean((predictions - y) ** 2)
            mae = np.mean(np.abs(predictions - y))

            results[ticker] = {
                'mse': float(mse),
                'mae': float(mae),
                'last_price': float(last_price)
            }

            print(f"\nEvaluation Results for {ticker}:")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
        
        except Exception as e:
            print(f"Error evaluating model for {ticker}: {str(e)}")
            continue
    
    return results

if __name__ == "__main__":
    print("Began training model")
    model = train_model(numStocks=3, epochs=5)

    print("\nPerformance evaluation:")
    evaluationResults = evaluate_model(model, n_test_stocks=1)


        