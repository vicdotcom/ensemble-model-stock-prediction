# Ensemble Model Stock Prediction

**Combining Statistical and Deep Learning Techniques for Stock Price Prediction**

This repository contains the code and data behind for the MSc dissertation titled above. We assess whether stock price forecasts can be improved by combining a classic statistical model (ARIMA) with modern deep learning architectures. The full dissertation write-up can be found in this repo.

## The problem

Stock prices are notoriously hard to forecast as they are volatile, nonlinear, and non-stationary (their statistical behaviour keeps shifting over time). Two families of models are traditionally used to tackle this:

- **ARIMA** (AutoRegressive Integrated Moving Average) - a statistical model that is excellent at capturing short-term linear trends but struggles with nonlinear patterns.
- **LSTM** (Long Short-Term Memory) - a neural network that can model nonlinear behaviour and "remember" past information, but tends to lose context over very long sequences.

Prior research has combined these two into an **ARIMA-LSTM hybrid**, which generally beats either model alone. This study asks whether swapping LSTM for two more recent architectures can improve prediction performance even further:

- **xLSTM** - an extended LSTM with a larger memory cell, designed to retain information over much longer sequences.
- **Transformer** - the attention-based architecture behind modern LLMs, which looks at an entire sequence at once instead of step-by-step.

The resulting **ARIMA-xLSTM** and **ARIMA-Transformer** models are combined via a **stacking ensemble** with a Ridge regression "meta-learner" that learns how to best blend individual model predictions into a final forecast.


## Data

Fifteen years (2010-01-01 to 2025-08-01) of daily Open/High/Low/Close/Volume (OHLCV) data for three NSE-listed stocks:

| Stock | Company | Folder |
|---|---|---|
| SCOM | Safaricom PLC | [`SCOM/`](SCOM) |
| KEGN | Kenya Electricity Generating Company PLC | [`KEGN/`](KEGN) |
| EQTY | Equity Group Holdings | [`EQTY/`](EQTY) |

From the raw OHLCV data, 92 candidate features were engineered per stock - technical indicators (moving averages, RSI, MACD, etc.) plus lagged price variables. These were narrowed down to a handful of final predictors per stock through a three-step selection process:

1. Correlation filtering (≥ 0.90 with closing price)
2. XGBoost-based Recursive Feature Elimination
3. Feature importance-score cutoff (≥ 0.02).

Each stock's analysis - cleaning, feature engineering, ARIMA/LSTM/xLSTM/Transformer training, and ensembling - is worked through in its own notebook:

- [`SCOM/Stock Prediction (SCOM).ipynb`](SCOM/Stock%20Prediction%20%28SCOM%29.ipynb)
- [`KEGN/Stock Prediction (KEGN).ipynb`](KEGN/Stock%20Prediction%20%28KEGN%29.ipynb)
- [`EQTY/Stock Prediction (EQTY).ipynb`](EQTY/Stock%20Prediction%20%28EQTY%29.ipynb)

<!-- Trained model artifacts are saved under each stock's `Models/` folder. -->

## Methodology, in brief

1. **Prepare the data** - clean outliers, engineer features, scale everything to [0, 1] with Min-Max scaling, and slice the series into sliding windows of 60 trading days (roughly one fiscal quarter) to predict the next day's close.
2. **Fit standalone models** per stock - ARIMA (orders selected via AIC), LSTM, xLSTM, and a Transformer, each tuned with Bayesian hyperparameter optimization (TPE).
3. **Build ensembles** - ARIMA-LSTM, ARIMA-xLSTM, and ARIMA-Transformer, each combined via a Ridge regression meta-learner.
4. **Evaluate** all seven models per stock on a 15-day out-of-sample test window using MSE, RMSE, MAE, and MAPE.

## Results

| Stock | Best model | MAPE |
|---|---|---|
| SCOM | ARIMA-LSTM | 0.74% |
| KEGN | ARIMA-Transformer | 1.73% |
| EQTY | LSTM | 0.37% |

The headline finding: **ensembling helps, but only when its neural network component is already performing well.** When the neural base model (xLSTM or Transformer) struggled, its errors carried into, and were sometimes amplified by, the ensemble. 

xLSTM, despite being the most architecturally advanced model tested, was consistently the least reliable. The likely culprit is a mismatch between its very large memory capacity (designed for language models trained on billions of tokens) and the comparatively small, noisy dataset available here (~2,600 training windows of individual, volatile NSE stocks) - a recipe for overfitting rather than genuine pattern learning. The Transformer fared better but was thrown off by how non-stationary stock prices are: attention weights learned on one period didn't always transfer to the next.

**Bottom line:** simpler models held up best. The study recommends **LSTM as a strong baseline** and **ARIMA-LSTM** as the preferred model where a hybrid is wanted, since it consistently blended ARIMA's linear stability with LSTM's nonlinear responsiveness without the overfitting or instability seen in the xLSTM/Transformer ensembles.

Full per-stock error tables, Ridge meta-learner weights, and prediction plots are in Chapter 5 of the dissertation write up.