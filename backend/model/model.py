import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

        return np.array(X), np.array(y), df['Close'][-1]

class StockPredictor:
    def __init__(self, input_shape=(20, 12), units=128):
        self.model = self._build_model(input_shape, units)
        
    def _build_model(self, input_shape, units):
        model = Sequential([
            LSTM(units=units, 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(0.2),
            
            LSTM(units=units//2, 
                 return_sequences=False),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        return self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath):
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        loaded_model = tf.keras.models.load_model(filepath)
        instance = cls(input_shape=loaded_model.input_shape[1:])
        instance.model = loaded_model
        return instance

def predict_stock_price(ticker, model, preprocessor):
    try:
        X, _, last_price = preprocessor.prepare_data(ticker)
        last_sequence = X[-1:]
        
        predicted_scaled = model.predict(last_sequence)
        predicted_price = preprocessor.scaler.inverse_transform(
            predicted_scaled.reshape(-1, 1)
        )[0][0]
        
        return {
            'ticker': ticker,
            'last_price': float(last_price),
            'predicted_price': float(predicted_price),
            'predicted_change': float((predicted_price - last_price) / last_price * 100)
        }
    
    except Exception as e:
        raise Exception(f"Error predicting stock price: {str(e)}")
