import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller

# Load dataset
df = pd.read_csv('data/ts2024.csv', parse_dates=['DateTime'])

# Display initial info
print("Initial dataset info:")
print(df.info())
print(df.describe())

# Check for missing and zero values
missing = df.isna().sum()
zeros = (df['X'] == 0).sum()
print(f"Missing values:\n{missing}")
print(f"Zero values in 'X': {zeros}")

# Replace zero values with 24-hour median
def replace_zeros_with_moving_median(series, window=24):
    zero_indices = series[series == 0].index
    for idx in zero_indices:
        left = max(0, idx - window // 2)
        right = min(len(series), idx + window // 2)
        median_val = series[left:right].median()
        series.at[idx] = median_val
    return series

df['X'] = replace_zeros_with_moving_median(df['X'])

# Handle outliers: replace > 99th percentile with median
threshold = df['X'].quantile(0.99)
df.loc[df['X'] > threshold, 'X'] = df['X'].median()

# Plot the time series
plt.figure(figsize=(15, 4))
plt.plot(df['DateTime'], df['X'])
plt.title("Time Series Plot")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# Split into 24 hourly series
hourly_series = {h: df[df['Hour'] == h]['X'].reset_index(drop=True) for h in range(24)}

# Apply Box-Cox transformation and ADF test
lambdas = {}
stationary_results = {}

for h, series in hourly_series.items():
    # Ensure positive values
    transformed, lmbda = boxcox(series + 1e-5)
    lambdas[h] = lmbda
    adf_stat, p_val, _, _, critical_vals, _ = adfuller(transformed)
    stationary_results[h] = {
        'ADF Statistic': adf_stat,
        'p-value': p_val,
        'Is Stationary': adf_stat < critical_vals['5%']
    }

# Summarize stationarity
print("Box-Cox Lambdas:")
print(lambdas)
print("\nADF Stationarity Test Results:")
for hour, result in stationary_results.items():
    print(f"Hour {hour}: {result}")

# Save processed data
df.to_csv("preprocessed_time_series.csv", index=False)