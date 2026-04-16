import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib


class LSTMModel:
    """LSTM model for time series prediction."""
    def __init__(self, seq_length: int = 60, n_features: int = 1,
                 lstm_units: int = 50, dropout: float = 0.2,
                 learning_rate: float = 0.001):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
        self.build_model(lstm_units, dropout, learning_rate)
    
    def build_model(self, lstm_units: int, dropout: float, learning_rate: float):
        self.model = Sequential([
            LSTM(lstm_units, return_sequences=True, 
                 input_shape=(self.seq_length, self.n_features)),
            Dropout(dropout),
            LSTM(lstm_units // 2, return_sequences=False),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i+self.seq_length])
            y_seq.append(y[i+self.seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X, y, epochs: int = 50, batch_size: int = 32,
            validation_split: float = 0.2, verbose: int = 1):
        X_seq, y_seq = self.create_sequences(X, y)
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return history
    
    def predict(self, X):
        X_seq = X[-self.seq_length:] if len(X) >= self.seq_length else \
                np.pad(X, ((self.seq_length - len(X), 0), (0, 0)))
        X_seq = X_seq.reshape(1, self.seq_length, self.n_features)
        return self.model.predict(X_seq, verbose=0)[0, 0]
    
    def save(self, filepath: str):
        self.model.save(filepath)
    
    def load(self, filepath: str):
        self.model = tf.keras.models.load_model(filepath)