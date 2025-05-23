

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

# Load and preprocess the data
file_path = "C:/Users/nosfe/time series/data/ts2024.csv"
df = pd.read_csv(file_path)
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
df = df.dropna(subset=['DateTime'])
df.set_index('DateTime', inplace=True)
df.drop(columns=['Date'], inplace=True)
df['X'] = df['X'].interpolate(method='time')

# Add time-based features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month

# Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['X', 'hour', 'dayofweek', 'month']])

# Create sequences
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length]
        y = data[i+sequence_length][0]  # predict 'X' column
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 168  # 1 week of hourly data
X, y = create_sequences(scaled_data, sequence_length)

# Split into training and testing sets
split = int(len(X) * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# Predict
y_pred = model.predict(X_test)

# Invert only 'X' scaling
x_scaler = MinMaxScaler()
x_scaler.fit(df[['X']])
y_test_inv = x_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = x_scaler.inverse_transform(y_pred)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Improved LSTM Time Series Prediction')
plt.legend()
plt.show()
