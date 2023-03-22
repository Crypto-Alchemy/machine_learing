import matplotlib.pyplot as plt
from learning.OHLCV import OHLCV
from learning.preprocess import preprocess
from learning.marketSentiment import marketSentiment
from learning.marketIndicators import addMarketIndicators
from learning.data_split import split_data
from learning.model_train import build_model, train_model, test_model
import datetime
import numpy as np
import pandas as pd

# Set up the input parameters
symbol = 'BTC/USD'
timeframe = '15m'
since = int((datetime.datetime.now() -
            datetime.timedelta(days=90)).timestamp() * 1000)
limit = 10000000

# Retrieve and preprocess the data
ohlcv = OHLCV(symbol, timeframe, since, limit)
df = preprocess(ohlcv)
df['day_of_week'] = df.index.dayofweek
df['hour_of_day'] = df.index.hour

# Split the data into training and testing sets
train_set, test_set = split_data(df, test_size=0.2)

# Train the LSTM model
model, scaler = train_model(train_set)

# Test the LSTM model
scaled_data = scaler.transform(test_set)
x_test = []
y_test = []

for i in range(60, len(scaled_data)):
    x_test.append(scaled_data[i - 60:i, 0])
    y_test.append(scaled_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = np.reshape(predictions, (predictions.shape[0], 1))
predictions = scaler.inverse_transform(predictions)


# Plot the predictions against the actual prices
plt.plot(predictions, label='Predicted Price')
plt.plot(test_set['close'].values, label='Actual Price')
plt.legend()
plt.show()

print(predictions)
