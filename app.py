import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# UI: Title and company selection
st.title("AI-Based Stock Price Prediction")
company = st.selectbox("Select a Company", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"])

# Download data
data = yf.download(company, start='2022-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
data = data[['Close']].dropna()

# Preprocess
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

look_back = 60
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(units=50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# Predict
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y.reshape(-1, 1))

# Show comparison chart
st.subheader("Actual vs Predicted")
fig1, ax1 = plt.subplots()
ax1.plot(actual, label="Actual")
ax1.plot(predicted, label="Predicted")
ax1.legend()
st.pyplot(fig1)

# Forecast next 10 days
future_days = 10
last_seq = X[-1]
future_preds = []

for _ in range(future_days):
    next_pred = model.predict(np.expand_dims(last_seq, axis=0))[0][0]
    future_preds.append(next_pred)
    last_seq = np.append(last_seq[1:], [[next_pred]], axis=0)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)

# Show future forecast
st.subheader(f"Next {future_days} Days Forecast")
fig2, ax2 = plt.subplots()
ax2.plot(future_dates, future_preds, color='green')
st.pyplot(fig2)
