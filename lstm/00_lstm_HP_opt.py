import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import ParameterGrid

# ------------------------------ LOAD DATA ------------------------------
file_path = os.path.join("data", "ts2024.csv")
df = pd.read_csv(file_path)

df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
df = df.dropna(subset=['DateTime'])                      # <- ✅ Drop bad timestamps
df = df.sort_values('DateTime')
df.set_index('DateTime', inplace=True)
df.drop(columns=['Date', 'Hour'], inplace=True, errors='ignore')

# ------------------------------ PREPROCESSING ------------------------------
n_missing = df['X'].isna().sum()
print(f"Missing values to forecast: {n_missing}")


print(df)
exit(0)

df_interp = df.copy()
df_interp['X'] = df_interp['X'].interpolate(method='time')
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_interp[['X']])

# ------------------------------ CREATE SEQUENCES ------------------------------
def make_sequences(data, seq_len, gap):
    X, y = [], []
    for i in range(seq_len, len(data) - gap):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# ------------------------------ HYPERPARAMETER TUNING ------------------------------
def build_model(seq_len, n_units, dropout):
    model = Sequential([
        LSTM(n_units, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(dropout),
        LSTM(n_units // 2),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

param_grid = {
    'seq_len': [24],#, 48, 72],        
    'n_units': [64],#, 128],
    'dropout': [0.1]#, 0.2]
}

best_val_loss = float('inf')
best_params = None
best_model = None

for params in ParameterGrid(param_grid):
    seq_len = params['seq_len']
    n_units = params['n_units']
    dropout = params['dropout']

    print(f"Trying: SEQ_LEN={seq_len}, UNITS={n_units}, DROPOUT={dropout}")

    X_all, y_all = make_sequences(scaled, seq_len, n_missing)
    n_val = int(0.1 * len(X_all))

    X_train, y_train = X_all[:-n_val], y_all[:-n_val]
    X_val, y_val     = X_all[-n_val:],  y_all[-n_val:]

    model = build_model(seq_len, n_units, dropout)

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0,
              validation_data=(X_val, y_val), callbacks=[es])

    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Val loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = params
        best_model = model

print(f"\n✅ Best params: {best_params}, Val Loss: {best_val_loss:.6f}")

# ------------------------------ FINAL TRAINING ------------------------------
final_seq_len = best_params['seq_len']
X_final, y_final = make_sequences(scaled, final_seq_len, n_missing)

best_model = build_model(**best_params)
best_model.fit(X_final, y_final, epochs=20, batch_size=32, verbose=1)

# ------------------------------ FORECAST MISSING ------------------------------
last_seq = scaled[-n_missing - final_seq_len:-n_missing].copy()
forecast_scaled = []

seq = last_seq
for _ in range(n_missing):
    pred = best_model.predict(seq.reshape(1, final_seq_len, 1), verbose=0)[0, 0]
    forecast_scaled.append(pred)
    seq = np.append(seq[1:], [[pred]], axis=0)

forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

# ------------------------------ PLOT ------------------------------
future_index = df[df['X'].isna()].index
plt.figure(figsize=(14, 4))
plt.plot(df_interp.index, df_interp['X'], label='Interpolated')
plt.plot(future_index, forecast, 'r-', label='LSTM Forecast')
plt.legend(); plt.tight_layout(); plt.show()

# ------------------------------ EXPORT ------------------------------
output = pd.DataFrame({
    'DateTime': future_index,
    'ARIMA': np.nan,
    'UCM': np.nan,
    'ML': forecast
})
output.to_csv("predictions_lstm_tuned.csv", index=False)
print(f"→ predictions_lstm_tuned.csv written with {len(output)} rows")
