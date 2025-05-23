"""
python lstm.py
Currently dropping test dataset as missing values
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os

file_path = "C:/Users/nosfe/time series/data/ts2024.csv"
 #os.path.join("data", "ts2024.csv")
df = pd.read_csv(file_path)
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
df = df.dropna(subset=['DateTime'])
df.set_index('DateTime', inplace=True)
df.drop(columns=['Date'], inplace=True)
df['X'] = df['X'].interpolate(method='time')

# Feature Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['X']])

# Create sequences
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 48  # e.g., 2 days of hourly data
X, y = create_sequences(scaled_data, sequence_length)

# Train/test split
split = int(len(X) * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.3)

# Predict
y_pred = model.predict(X_test)

# Invert scaling
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Plot predictions vs actuals
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('LSTM Time Series Prediction')
plt.legend()
plt.show()