# Hourly Time Series Forecasting of Traffic Congestion

This repository presents a comprehensive time series analysis of a traffic congestion indicator measured hourly on a major U.S. freeway. The goal is to forecast traffic levels using statistical and deep learning models that handle strong daily and weekly seasonality.

Three modeling families are explored and compared:

- **SARIMA/SARIMAX**: Seasonal ARIMA models
- **UCM**: Unobserved Components Models
- **Deep Learning**: LSTM/GRU networks

---

## Dataset

The dataset consists of 17,544 hourly observations of a congestion indicator:

| Column     | Description                                |
|------------|--------------------------------------------|
| DateTime   | Full timestamp (`yyyy-mm-dd hh:mm:ss`)     |
| Date       | Calendar date (`yyyy-mm-dd`)               |
| Hour       | Hour of day (0–23)                         |
| X          | Target congestion value (float)            |

744 missing values (future hours) are present and serve as the forecasting targets.

---

## Exploratory analysis

- **Stationarity**: Verified using the Augmented Dickey-Fuller test (statistic: −15.711, p-value ≈ 0)
- **Seasonality**:
  - Strong **daily (24h)** and **weekly (168h)** seasonal effects
  - Observed via ACF/PACF and calendar heatmaps

---

## Training pipeline

Train → Validation → Test
    |        |         └── 744 points (target)
    |        └── 744 points
    └── Remaining history

The train and validation are used to select the optimal model for every family. The candidate model is then retrained (with the optimal configuration) on the training set plus the validation and then it forecasts the test set.

---

## Results

### SARIMAX

| Model Type                        | Validation MAE |
|----------------------------------|----------------|
| ARIMA(2,0,1)                     | 0.0313         |
| SARIMAX(1,0,1)(1,0,1)[24]        | 0.0186         |
| **Per-Hour SARIMAX (D adjusted)**| **0.0120**     |

This approach uses per-hour sub-series and applies seasonal differencing conditionally for evening hours. Weekly seasonality (period = 7 days) is modeled explicitly.

---

### Unobserved Components Model (UCM)

| Model Type                         | Validation MAE |
|-----------------------------------|----------------|
| Global UCM                        | 0.0274         |
| Per-Hour UCM                      | 0.0251         |
| **Per-Hour UCM + Holiday Regressor** | **0.0125**     |

- Components used: local linear trend + weekly seasonality
- Holiday indicator significantly improved accuracy

---

### Deep Learning (LSTM / GRU)

| Architecture                      | Validation MAE (scaled) |
|----------------------------------|--------------------------|
| LSTM (1x64)                      | 0.0179                   |
| LSTM (2x128 + dropout)          | 0.0179                   |
| GRU (2x64)                      | 0.0180                   |
| **LSTM (1x128) + calendar features** | **0.0178**           |

Key features:
- Lookback window: 168 hours
- Forecast horizon: 1 hour ahead
- Added calendar encodings (hour, weekday, month) improved performance

---

## Model comparison

| Model Type  | Best Configuration                  | Validation MAE |
|-------------|--------------------------------------|----------------|
| SARIMAX     | Per-Hour + Conditional Differencing | **0.0120**     |
| UCM         | Per-Hour + Holiday Regressor        | 0.0125         |
| Deep Learning | LSTM + Calendar Features           | 0.0178         |

---

## Future perspectives

1. Add external regressors (e.g., weather, local events)
2. Ensemble methods (e.g., stacking SARIMA + DL)
3. Hyperparameter tuning for SARIMA and LSTM
4. Explore advanced DL: Transformers, TCN, hybrid models
5. Try modeling level shifts, deterministic trends in UCM