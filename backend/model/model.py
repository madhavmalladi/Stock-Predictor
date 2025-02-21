import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class StockDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.sequence_length = 20

    def calculate_technical_indicators(self, df):
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

        df['Volume_SMA_7'] = df['Volume'].rolling(window=7).mean()

        df['Price_Change'] = df['Close'].diff()
        df['Price_PCT_Change'] = df['Close'].pct_change()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        relative_strength = gain/loss
        df['RSI'] = 100 - (100 / (1 + relative_strength))

        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2*df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2*df['Close'].rolling(window=20).std()

        return df
    
    def prepare_data(self, ticker, period='1y'):
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df = self.calculate_technical_indicators(df)

        features = [
            'Close', 'Volume', 'SMA_7', 'SMA_20', 'SMA_50', 'Daily_Return', 
            'Volatility', 'Volume_SMA_7', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower'
        ]
    
        data = df[features].dropna()
        scaled_data = self.scaler.fit_transform(data)

        X = []
        y = []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, 0])

        return torch.FloatTensor(X), torch.FloatTensor(y), df['Close'][-1]

class StockPredictor(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, dropout=0.2):
        super(StockPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        last_time_step = lstm_out[:, -1, :]
        out = self.fc1(last_time_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)

        return out
    
def predict_stock_price(ticker, model, preprocessor):
    try:
        X, _, last_price = preprocessor.prepare_data(ticker)
        last_sequence = X[-1].unsqueeze(0)
        model.eval()
        with torch.no_grad():
            predicted_scaled = model(last_sequence)
        
        predicted_price = preprocessor.scaler.inverse_transform(
            predicted_scaled.cpu().numpy().reshape(-1, 1)
        )[0][0]
        
        return {
            'ticker': ticker,
            'last_price': float(last_price),
            'predicted_price': float(predicted_price),
            'predicted_change': float((predicted_price - last_price) / last_price * 100)
        }
    
    except Exception as e:
        raise Exception(f"Error predicting stock price: {str(e)}")
