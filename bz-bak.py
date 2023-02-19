import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import os
from datetime import datetime
  
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('./ETH-USD.csv')
print(data.shape)
print(data.sample(7))

data.info()

data['date'] = pd.to_datetime(data['date'])
data.info()

data['date'] = pd.to_datetime(data['date'])

apple = data
prediction_range = apple.loc[(apple['date'] > datetime(2013,1,1))
& (apple['date']<datetime(2018,1,1))]
plt.plot(apple['date'],apple['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Ethereum USD Stock Prices")
plt.show()

close_data = apple.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)

from sklearn.preprocessing import MinMaxScaler
  
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
  
train_data = scaled_data[0:int(training), :]
# prepare feature and labels
x_train = []
y_train = []
  
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
  
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary

model.compile(optimizer='adam',
              loss='mean_squared_error')
history = model.fit(x_train,
                    y_train,
                    epochs=10)

test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
  
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  
# predict the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
  
# evaluation metrics
mse = np.mean(((predictions - y_test) ** 2))
print("MSE", mse)
print("RMSE", np.sqrt(mse))

train = apple[:training]
test = apple[training:]
test['Predictions'] = predictions
  
train = train.set_index('date')
test = test.set_index('date')

plt.figure(figsize=(10, 8))
plt.plot(train['close'])
plt.plot(test[['close', 'Predictions']])
plt.title('Ethereum USD Stock Close Price')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()

# now predict 1 week from now 

# Get the last 60 days of the "close" column
last_60_days = dataset['close'][-60:].values

# Scale the last 60 days of data
last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

# Reshape the data to have shape (1, 60, 1)
X_test = last_60_days_scaled.reshape((1, 60, 1))

# Use the model to make a prediction for the next value in the time series
predicted_price = model.predict(X_test)

# Inverse scale the predicted value
predicted_price = scaler.inverse_transform(predicted_price)

# Print the predicted price
print(predicted_price)

