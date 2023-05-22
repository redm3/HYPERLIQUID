import numpy as np
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from time import sleep
import ccxt

exchange = ccxt.binance()

symbol = 'BTC/USD'

model = load_model('btc_model7.h5')

ticker = exchange.fetch_ticker(symbol)

current_price = ticker['last']
current_price = np.array(current_price).reshape(-1,1)



# Load training data
training_data = pd.read_csv(r'C:/Users/macmw/Documents/GitHub/HyperLiquid/BTC-USD-actual.csv', usecols=['close'])
# Convert dataframe to numpy array
training_data = training_data.values

# Define the sequence length
sequence_length = 60

# Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(training_data)
last_60_bars = training_data[-sequence_length:]
last_60_bars_scaled = scaler.transform(last_60_bars)
X_test = []
X_test.append(last_60_bars_scaled)
X_test = np.array(X_test)
# The input shape should be (None, sequence_length, 1)
X_test = np.reshape(X_test, (X_test.shape[0], sequence_length, 1))

pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price_value = pred_price
print(pred_price_value)

if pred_price > current_price:
    print(f'${pred_price} > ${current_price} - Buy long = True!')
    long= True
elif pred_price < current_price:
    print(f'${pred_price} < ${current_price} - Buy long = False!')
    long = False