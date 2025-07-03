import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import datetime

# 1. Download stock data (dynamic up-to-date)
ticker = 'GOOGL'  # You can change this to 'AAPL', 'TSLA', etc.
today = datetime.datetime.today().strftime('%Y-%m-%d')
data = yf.download(ticker, start='2022-01-01', end=today)

# 2. Preprocess
data = data[['Close']]
data.dropna(inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 3. Create sequences for training
look_back = 60
X, y = [], []
for i in range(look_back, len(data_scaled)):
    X.append(data_scaled[i - look_back:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Train the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# 5. Predict on training set for comparison
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transform(y.reshape(-1, 1))

# 6. Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.title(f'{ticker} Stock Price Prediction (Up to {today})')
plt.plot(real_prices, color='blue', label=f'Actual {ticker} Price')
plt.plot(predicted_prices, color='red', label=f'Predicted {ticker} Price')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# 7. Predict next 10 days
future_days = 10
last_sequence = X[-1]
future_predictions = []

for _ in range(future_days):
    next_pred = model.predict(np.expand_dims(last_sequence, axis=0))[0][0]
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

# 8. Plot future predictions
plt.figure(figsize=(10,5))
plt.plot(future_dates, future_predictions, color='green', label='Future Predictions (Next 10 Days)')
plt.xlabel("Date")
plt.ylabel("Predicted Price (USD)")
plt.title(f'{ticker} Future Stock Price Forecast')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
