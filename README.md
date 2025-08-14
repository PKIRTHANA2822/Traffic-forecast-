# Project Title
- Time Series Forecasting with Prophet (formerly fbprophet)
![1700221887php9lV4wW_480x360](https://github.com/user-attachments/assets/b4a866f1-86c7-4b2a-b514-ddf1ceb8b280)
# Objective
- Build a robust, explainable time series model using Prophet to forecast future values and quantify uncertainty. The model should capture trend, seasonality (daily/weekly/yearly), holiday effects, and optional external drivers (regressors) to support planning and decision‑making.
# Why We Use This Project (Use Cases)
- Demand/Sales forecasting: inventory planning, staffing, budgeting.
- Traffic/Engagement forecasting: server capacity, campaign scheduling.
- Finance/Operations: revenue projections, cost forecasting, anomaly detection.
- Education/EdTech analytics: forecast enrollments, activity, or assessment volumes.
- Prophet is chosen because it:
- Handles missing data, trend changes (changepoints), and outliers gracefully.
- Is additive/explainable with interpretable components.
- Requires minimal tuning to get a strong baseline.
# Data & Assumptions
- Input time series with two columns: ds (datetime) and y (numeric target).
- Frequency is regular (e.g., daily/weekly/hourly). If not, resample to a consistent cadence.
- Optional external regressors (e.g., promotions, price, weather) and holiday calendars.
# Step‑by‑Step Approach
- Load & Inspect
- Read raw data, check types, ensure ds is timezone‑aware or consistently localized.
# Exploratory Data Analysis (EDA)
- Line plot of y over time; detect trends, seasonality, outliers.
- Decompose (optional) to visualize trend/seasonality; check stationarity only for context.
- Preprocessing
- Sort by date; handle duplicates; impute/forward‑fill missing periods if needed.
- Optional transform (e.g., log) if variance grows with level.
# Feature Engineering
- Holidays/events: add country holidays; add custom event windows (pre/post effects).
- Regressors: add exogenous features (promo flags, price, weather), properly lagged if needed.
- Seasonalities: add extra seasonality (e.g., monthly, quarterly) and choose additive vs. multiplicative mode.
# Model Training
- Configure Prophet (growth, changepoints, priors, seasonality mode) and fit.
- Model Testing/Validation
- Time‑series cross‑validation using Prophet’s cross_validation & performance_metrics (MAPE, MAE, RMSE, sMAPE).
- Visual diagnostics of forecast error by horizon.
# Forecasting & Reporting
- Generate future periods; plot forecast & components; export tables/graphics.
- Iteration/Tuning
- Adjust changepoint prior, seasonality prior, holiday prior, and added seasonalities/regressors.
- Exploratory Data Analysis (Checklist)
- Structure: df.info(), date range, frequency gaps, duplicates.
- Missing values: quantify and decide on imputation/resampling.
- Outliers: visualize spikes/dips; mark with y_origin and capped/cleaned y if needed.
- Seasonality: weekly/yearly patterns; weekday/weekend/holiday patterns.
- Trend changes: structural breaks (policy changes, launches, semester starts, etc.).
# Quick EDA snippets
ax = df.set_index('ds')['y'].plot(figsize=(12,4), title='Target over time')
ax.set_xlabel('Date'); ax.set_ylabel('y')
# Feature Selection
- Core: ds, y.
- Optional regressors: choose variables with clear causal/leading relationships.
- Validate usefulness via domain logic and lagged correlation.
- Avoid leaky features (future info).
# Feature Engineering
- Holidays:
- from prophet import Prophet
- m = Prophet(weekly_seasonality=True, yearly_seasonality=True, seasonality_mode='additive')
# Example: India holidays (set your country)
m.add_country_holidays(country_name='IN')
Custom events:
import pandas as pd
specials = pd.DataFrame({
    'holiday': 'promo',
    'ds': pd.to_datetime(['2024-11-10','2025-02-14']),
    'lower_window': -2,
    'upper_window': 2,
})
m = Prophet(holidays=specials)
# Additional seasonalities:
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
# Multiplicative effects if variance scales with level
m = Prophet(seasonality_mode='multiplicative')
Regressors:
# df must contain these columns
for col in ['price','promo_flag','rain_mm']:
    m.add_regressor(col)
Model Training

# Ensure columns
ts = raw_df.rename(columns={'timestamp':'ds', 'value':'y'})[['ds','y']].copy()
ts['ds'] = pd.to_datetime(ts['ds'])
ts = ts.sort_values('ds')

# (Optional) log transform for positive series
# ts['y'] = np.log1p(ts['y'])
m = Prophet(
    growth='linear',
    seasonality_mode='additive',  # or 'multiplicative'
    changepoint_prior_scale=0.05, # increase for more flexibility
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    weekly_seasonality=True,
    yearly_seasonality=True,
)
# Visualization
from prophet.plot import plot_plotly, plot_components_plotly
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
Model Testing (Time‑Series Cross‑Validation)
from prophet.diagnostics import cross_validation, performance_metrics
# Define CV windows (example for daily data)
cv_df = cross_validation(
    m,
    initial='365 days',  # initial training window
    period='30 days',    # step between cutoffs
    horizon='90 days'    # forecast horizon to evaluate
)
metrics = performance_metrics(cv_df)
metrics[['horizon','mape','smape','mae','rmse']].head()
# Visual diagnostics
from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(cv_df, metric='mape')
# Output / Deliverables
<img width="748" height="541" alt="Screenshot 2025-08-14 122505" src="https://github.com/user-attachments/assets/ffe8d2c6-8f89-492b-b54f-09e3518e6a81" />

