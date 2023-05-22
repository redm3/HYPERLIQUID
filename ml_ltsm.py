#we built an LSTM model that predicts the price of BTC/USDon
#on the 15min chart and then we want to output 
#a signal at the end to ouyr hyperliquid algo

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import schedule
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#when I wanna load a model
#from keras.models import load_model
#load the saved model from the file
#loaded_model = load_model("btc_model.h5")

df = pd.read_csv('C:/Users/macmw/Documents/GitHub/HyperLiquid/BTC-USD-actual.csv')
print(df)

print(df.shape)

#visualize close
plt.figure(figsize=(16,8))
plt.title('Close Prices')
plt.plot(df['close'])
plt.xlabel('datetime',fontsize=16)
plt.ylabel('Close price',fontsize=16)

data = df.filter(['close'])
dataset = data.values
training_data_len = math.ceil(len(dataset)* .8)
training_data_len

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(50, return_sequences =True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=50)
model.save("btc_model7.h5")

test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

train = data[:training_data_len]
valid = data[training_data_len:].copy()
valid['predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('model')
plt.xlabel('datetime',fontsize=16)
plt.plot(train['close'])
plt.plot(valid[['close','predictions']])
plt.legend(['train', 'val', 'predictions'], loc= 'lower right')

print(valid)

btc_quote = pd.read_csv('C:/Users/macmw/Documents/GitHub/HyperLiquid//BTC-USD-actual.csv')
new_df = btc_quote.filter(['close'])
last_300_bars = new_df[-300:].values
last_300_bars_scaled = scaler.transform(last_300_bars)
X_test = []
X_test.append(last_300_bars_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
print(pred_price[0][0])

