import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.structural import UnobservedComponents

# ------------------------------ LOAD DATA ------------------------------
file_path = os.path.join("data", "ts2024.csv")
df = pd.read_csv(file_path)

df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
df = df.dropna(subset=['DateTime'])
df = df.sort_values('DateTime')
df.set_index('DateTime', inplace=True)
df.drop(columns=['Date', 'Hour'], inplace=True, errors='ignore')

# ------------------------------ PREPROCESSING ------------------------------
n_missing = df['X'].isna().sum()
print(f"Missing values to forecast: {n_missing}")

df_train = df[~df['X'].isna()].copy()
df_train = df_train[~df_train.index.duplicated(keep='first')]
df_train = df_train.asfreq("h")  # Ensure regular hourly time series

# ------------------------------ BUILD & FIT MODEL ------------------------------
model = UnobservedComponents(
    endog=df_train['X'],
    level='local linear trend',
    seasonal=24
)
results = model.fit(disp=False)
print(results.summary())

# ------------------------------ FORECAST ------------------------------
forecast_result = results.get_forecast(steps=n_missing)
forecast_mean = forecast_result.predicted_mean

# Construct datetime index manually
start = df_train.index[-1] + pd.Timedelta(hours=1)
forecast_index = pd.date_range(start=start, periods=n_missing, freq='h')
forecast_mean.index = forecast_index

# ------------------------------ PLOT ------------------------------
plt.figure(figsize=(14, 4))
plt.plot(df.index, df['X'], label='Observed')
plt.plot(forecast_mean.index, forecast_mean, 'g', label='UCM Forecast')
plt.legend(); plt.tight_layout(); plt.show()

# ------------------------------ EXPORT ------------------------------
output = pd.DataFrame({
    'DateTime': forecast_mean.index,
    'ARIMA': np.nan,
    'UCM': forecast_mean.values,
    'ML': np.nan
})
output.to_csv("predictions_ucm.csv", index=False)
print(f"→ predictions_ucm.csv written with {len(output)} rows")
