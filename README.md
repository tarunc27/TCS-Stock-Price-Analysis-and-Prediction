# TCS Stock Price Analysis and Prediction

A comprehensive project focused on analyzing and forecasting the stock prices of **Tata Consultancy Services (TCS)** using both traditional and deep learning models. This project demonstrates the application of time series analysis, feature engineering, and model comparison to build predictive systems for financial data.

---

## Tools & Technologies

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn (Linear Regression, Random Forest)  
- TensorFlow / Keras (LSTM)  
- yfinance (for data sourcing)  

---

## Dataset

- Source: Yahoo Finance (or local CSV)  
- Period: August 2002 – December 2023  
- Features: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits`

---

## Project Workflow

### 1. Data Collection
- Data imported from CSV or fetched live using `yfinance`.

### 2. Data Preprocessing
- Null values handled with forward fill.
- Columns converted to correct data types.
- Dataset sorted by date for time-series integrity.

### 3. Exploratory Data Analysis (EDA)
- Close price trend visualization
- Trading volume over time
- Moving averages (50 & 200-day)
- Correlation heatmap
- Daily percentage change distribution

### 4. Feature Engineering
- Lag features (`Prev_Close`)
- Rolling averages (`MA50`, `MA200`, `Moving_Avg_Close`)
- Temporal features: `Year`, `Month`, `Day`, `Day_of_Week`

### 5. Modeling

#### Linear Regression
- Simple regression using numerical and temporal features.
- Evaluation: MSE, MAE, R²

#### LSTM (Long Short-Term Memory)
- Sequence modeling with normalized data.
- Built using Keras Sequential API.
- Evaluation: MAE with actual vs. predicted plots.

#### Random Forest Regressor
- Nonlinear model for comparison and future use.
- Evaluation based on MSE.

---

## Results & Insights

- LSTM outperformed Linear Regression in sequential accuracy.
- Random Forest showed promise for nonlinear dependencies.
- Most influential features: `Open`, `High`, `Low`, and previous day prices.
- Daily price changes followed a normal-like distribution with rare outliers.

---

## Future Work

- Incorporate technical indicators (e.g., RSI, MACD)
- Add external signals (NIFTY index, news sentiment)
- Multi-step forecasting for extended prediction horizon
- Real-time deployment via Flask or Streamlit
- Portfolio-level forecasting with multiple stocks

---

## Outputs

- Plots: `close_price.png`, `volume.png`, `moving_averages.png`, etc.
- LSTM prediction CSV: `lstm_predictions.csv`
- Saved models: `TCS_Stock_Predictor.pkl`

---

## References

- [Yahoo Finance - TCS.NS](https://finance.yahoo.com/quote/TCS.NS)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [TensorFlow - LSTM Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.
- Brownlee, J. (2017). *Introduction to Time Series Forecasting with Python*. Machine Learning Mastery.

---

## Author

**TARUN C**  
Data Analyst | ML Developer  
LinkedIn: [your-link-here]  
Portfolio: [your-portfolio-link]
