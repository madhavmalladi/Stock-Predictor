import os
import pandas as pd
import numpy as np
import yfinance as yf
from model import StockPredictor, StockDataPreprocessor

def get_tickers(numStocks = 50):
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        tickers = table['Symbol'].tolist();
        marketCaps = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                marketCap = stock.info.get('marketCap', 0)
                marketCaps[ticker] = marketCap
            except:
                continue

        sortedTickers = sorted(marketCaps.items(), key=lambda x: x[1], reverse=True) 
        topTickers = [ticker for ticker, _ in sortedTickers][:numStocks] 

        print(f"Successfully fetched {len(topTickers)} tickers")
        return topTickers
    except Exception as e:
        print(f"Error getting tickers: {str(e)}")
        # returning major tickers in case of failure
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'V'] 
    
def train_model(numStocks = 50, epochs = 50, batch_size = 32, save_path = 'saved_models'):
    print("Getting stock tickers...")
    tickers = get_tickers(numStocks)
    print(f"Training on tickers: {', '.join(tickers)}")
    
    model = StockPredictor()
    preprocessor = StockDataPreprocessor()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for epoch in range(epochs):
        totalLoss = 0
        totalMae = 0
        successfulTickers = 0

        for ticker in tickers:
            try:
                print(f"\nTraining on {ticker} - Epoch {epoch+1}/{epochs}")
                X,y, _ = preprocessor.prepare_data(ticker)
                if len(X) < batch_size:
                    print(f"Not enough data for {ticker}. Skipped")
                    continue
                history = model.fit(X, y, epochs = 1, batch_size=batch_size, validation_split=0.2, verbose=1)
                totalLoss += history.history['loss'][0]
                totalMae += history.history['mae'][0]
                successfulTickers += 1
            except Exception as e:
                print(f"Error occured: {str(e)} for ticker {ticker}")
                continue

        if successfulTickers > 0:
            avgLoss = totalLoss / successfulTickers
            avgMae = totalMae / successfulTickers
            print(f'\nEpoch [{epoch+1}/{epochs}] Summary:')
            print(f'Stocks processed successfully: {successfulTickers}/{len(tickers)}')
            print(f'Average Loss: {avgLoss:.4f}')
            print(f'Average MAE: {avgMae:.4f}')
        
        if (epoch+1) % 10 == 0:
            checkpointPath = os.path.join(save_path, f'model_epoch_{epoch+1}.h5')
            model.save(checkpointPath)
            print(f"Checkpoint saved: {checkpointPath}")

    finalModelPath = os.path.join(save_path, 'final_model.h5')
    model.save(finalModelPath)
    print(f"Final model saved to {finalModelPath}")

    return model
    
def evaluate_model(model, n_test_stocks = 10):
    testTickers = get_tickers(n_test_stocks + 50)[-n_test_stocks:]
    preprocessor = StockDataPreprocessor()

    results = {}
    for ticker in testTickers:
        try:
            X, y, lastPrice = preprocessor.prepare_data(ticker, period = '6mo')
            predictions = model.predict(X)

            mse = np.mean((predictions - y.reshape(-1,1)) ** 2)
            mae = np.mean(np.abs(predictions-y.reshape(-1,1)))

            results[ticker] = {
                'mse': float(mse),
                'mae': float(mae)
            }

            print(f"\nEvaluation Results for {ticker}:")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
        
        except Exception as e:
            print(f"Error evaluating model for {ticker}: {str(e)}")
            continue
    
    return results

if __name__ == "__main__":
    print("Starting model training")
    model = train_model(numStocks = 50, epochs = 50)

    print("\nEvaluating model performance")
    evaluationResults = evaluate_model(model, n_test_stocks=10)


        