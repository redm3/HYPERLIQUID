from eth_account.signers.local import LocalAccount
import eth_account
import json
import time
import utils
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from hyperliquid.utils.types import UserEventsMsg
import pandas as pd
import datetime
import schedule
import requests
from sklearn.preprocessing import MinMaxScaler
from time import sleep
import numpy as np
from keras.models import load_model
import pandas as pd
import ccxt

exchange = ccxt.binance()
symbol = 'INJ'
timeframe = '1m'
limit = 100
max_tr = 300
no_trading_past_hrs = 24
timeframesma = '1h'
sma_period = 20
max_loss = -1
target = 5
size = 5
min_acct_value = 1
long_only = False
short_only = False
binance_symbol = symbol + '/USD'

def get_sz_decimals(symbols):
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type' : 'application/json'}
    data = {'type' : 'meta'}
    response = requests.post(url,headers=headers, data = json.dumps(data))
    if response.status_code == 200:
        data = response.json()
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
            return sz_decimals
        else:
            print('Symbol not found')
    else:
        print('Error', response.status_code)

def ask_bid(symbol):
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type' : 'application/json'}

    data = {
        "type": "l2Book",
        "coin" : symbol
    }

    response = requests.post(url,headers=headers, data = json.dumps(data))
    print(response.status_code)
    print(response.text)
    l2_data = response.json()
    print(l2_data)
    l2_data = l2_data['levels']
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[0][0]['px'])
    ask = float(ask)
    bid = float(bid)
    print(f'ask:{ask} bid: {bid}')

    return ask, bid, l2_data

def get_total_fills(wallet_address):
    headers = {'Content-Type' : 'application/json'}
    data = {"type":"userFills", "user":wallet_address}
    response = requests.post("https://api.hyperliquid.xyz/info",headers=headers, data = json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()
        df = pd.DataFrame(response_data)

        df['sz'] = df['sz'].astype(float).abs()
        df['px'] = df['px'].astype(float).abs()
        df['vol'] = (df['sz'] * df['px'])

        fills = response_data['data']['fills']

        # Use list comprehension to generate UTC times from 'fills' list
        utc_times = [pd.to_datetime(str(fill['time'])).tz_localize('UTC').strftime('%Y-%m-%d') for fill in fills]

        coins = [d['coin'] for d in fills]
        start_positions = [d['startPosition'] for d in fills]

        fills_df = pd.DataFrame({'time(UTC)':utc_times, 'coin':coins, 'startPosition':start_positions})

        total_start_position = fills_df['startPosition'].astype(float).sum()

        print(fills_df)
        print("Total startPosition", total_start_position)

        return total_start_position
    
    else:
        response.raise_for_status()

# This function converts a datetime object to epoch milliseconds
def datetime_to_epoch_ms(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return int((dt-epoch).total_seconds() * 1000.0)

# This function returns the start and end times in milliseconds for a range of minutes back from the current time
def get_time_range_ms(minutes_back):
    current_time_ms = int(datetime.datetime.utcnow().timestamp() * 1000)
    start_time_ms = current_time_ms - (minutes_back * 60 * 1000)
    end_time_ms = current_time_ms
    return start_time_ms, end_time_ms

def get_ohlcv(binance_symbol, timeframe='1h', limit=100):
    coinbase = ccxt.coinbasepro()

    ohlcv = coinbase.fetch_ohlcv(binance_symbol,timeframe ,limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df = df.tail(limit)

    df['support'] = df[:-2]['close'].min()
    df['resis'] = df[:-2]['close'].max()

    return df

def sma_signal(timeframesma = timeframesma, sma_period=sma_period):
    coinbase = ccxt.coinbasepro()

    ohlcv = coinbase.fetch_ohlcv(binance_symbol,timeframesma ,limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df.set_index('timestamp', inplace=True)
    sma = df['close'].rolling(window=sma_period).mean()

    current_price =  ask_bid(symbol)[0]
    current_price = float(current_price)

    if current_price > sma[-1] * 0.9975 and current_price < sma[-1] * 1.005: 
        signal = 1 if current_price > sma[-1] else 0  #long
    else:
        signal = None #short
    
    print(f"current price: {current_price:.2f}")
    print(f"1-hour SMA: {sma[-1]:.2f}")
    print(f"Signal:{signal}")

    return signal #1 for long 0 for short

sma_signal()

def supply_demand_zones(symbol, timeframe,limit):
    print('starting supply and demand zone calculations...')

    sd_limit = 24
    sd_sma = 20

    sd_df = pd.DataFrame()

    df = get_ohlcv(binance_symbol, timeframe, limit)

    supp_1h = df.iloc[-1]['support']
    resis_1h = df.iloc[-1]['resis']

    df['supp_lo'] = df[:-2]['low'].min()
    supp_lo_1h = df.iloc[-1]['supp_lo']

    df['res_hi'] = df[:-2]['low'].max()
    res_hi_1h = df.iloc[-1]['res_hi']

    sd_df['1h_dz'] = [supp_lo_1h, supp_1h]
    sd_df['1h_sz'] = [res_hi_1h, resis_1h]
    
    return sd_df

def limit_order(coin: str, is_buy: bool, sz: float, limit_px: float, reduce_only: bool):
    config = utils.get_config()
    account: LocalAccount = eth_account.Account.from_key(config["secret_key"])
    exchange = Exchange(account, constants.MAINNET_API_URL)
    sz = round(sz,1)
    limit_px = round(limit_px,1)
    print(f'placing limit order for {coin} {sz} @ {limit_px} ')
    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif":"Gtc"}},reduce_only) # include reduce_only here

    if is_buy == True:
        print(f"limit BUY order placed, resting: {order_result['response']['data']}")
    else:
        print(f"limit SELL order placed, resting: {order_result['response']['data']}")
    return order_result

def get_position():
    config = utils.get_config()
    account: LocalAccount = eth_account.Account.from_key(config["secret_key"])
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    print(f"This is current account value: {user_state['marginSummary']['accountValue']}") 
    positions = []
    for position in user_state["assetPositions"]:
        if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]):
            positions.append(position["position"])
            in_pos = True
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"])*100
            break
        else:
            in_pos = False
            size = 0
            pos_sym = None
            entry_px = 0
            pnl_perc = 0
    if size > 0:
        long = True
    elif size < 0:
        long = False
    else:
        long = None

    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long

# This function cancels all open orders on the HyperLiquid exchange.
def cancel_all_orders():
    config = utils.get_config()
    account: LocalAccount = eth_account.Account.from_key(config["secret_key"])
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    #user_state = info.user_state(account.address)
    #open_orders = user_state.get('openOrders', [])
    open_orders = info.open_orders(account.address)

    print('above are the open orders... need to cancel any...')
   # print(open_orders)  # Add this line to print out the structure of open_orders
    for open_order in open_orders:
        exchange.cancel(open_order["coin"], open_order["oid"])

from typing import List
from decimal import Decimal

def get_open_order_prices() -> List[Decimal]:
    config = utils.get_config()
    account: LocalAccount = eth_account.Account.from_key(config["secret_key"])
    api_url = constants.MAINNET_API_URL
    exchange = Exchange(account,api_url)
    info = Info(api_url, skip_ws=True)
    open_orders = info.open_orders(account.address)

    open_order_prices = []
    for open_order in open_orders:
        open_order_prices.append(Decimal(open_order['limitPx']))

    return open_order_prices

def kill_switch(symbol):
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position() #check here

    while im_in_pos == True:
        cancel_all_orders()

        askbid = ask_bid(pos_sym)
        ask = askbid[0]
        bid = askbid[1]

        pos_size = abs(pos_size)

        if long == True:
            limit_order(pos_sym, False, pos_size, ask, True) # Set reduce_only to True
            print('kill switch - SELL TO CLOSE SUBMITTED')

        elif long == False:
            limit_order(pos_sym, True, pos_size, bid, True) # Set reduce_only to True
            print('kill switch - BUY TO CLOSE SUBMITTED')

        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position()
        
    print('position successfully closed in kill switch')

def pnl_close():
    print('entering pnl close')
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position()
    pnl = pnl_perc * pos_size # calculate actual profit or loss
    if pnl >= target:
        print(f'pnl gain is {pnl} and reached target of {target}... closing pos')
        kill_switch(pos_sym)
    elif pnl <= -max_loss:
        print(f'pnl loss is {pnl} and exceeded max loss of {-max_loss}... closing pos')
        kill_switch(pos_sym)
    else:
        print(f'pnl gain is {pnl} but has not yet reached target of {target} or exceeded max loss of {-max_loss}... not closing pos')
    print('finished with pnl close')

bars = get_ohlcv(binance_symbol, timeframe='1h', limit=100)

def tr(data):
    data['previous_close'] = data['close'].shift(1)
    data['high-low'] = abs(data['high']-data['low'])
    data['high-pc'] = abs(data['high']-data['previous_close'])
    data['low-pc'] = abs(data['low']-data['previous_close'])
    tr = data[['high-low','high-pc', 'low-pc']].max(axis=1)
    return tr

def atr(data,period):
    data['tr'] = tr(data)
    atr = data['tr'].rolling(period).mean()
    return atr

def no_trading(data,periood):
    data[no_trading] = (data['tr'] > max_tr).any()
    no_trading = data['no_trading']
    return no_trading
 
def get_atr_notrading():
    bars = get_ohlcv('BTC/USD', timeframe='1h')
    df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    atrr = atr(df,7)
    print(atrr)
    df['no_trading'] = (df['tr'] > max_tr).any()

    no_trading = df['no_trading'].iloc[-1]
    print(df)

    print(f'no trading{no_trading}')

    return no_trading
def bot():
    sdz = supply_demand_zones(symbol, timeframe, limit)

    sz_1hr = sdz['1h_sz']
    sz_1hr_0 = sz_1hr.iloc[0]
    sz_1hr_1 = sz_1hr.iloc[-1]

    dz_1hr = sdz['1h_dz']
    dz_1hr_0 = dz_1hr.iloc[0]
    dz_1hr_1 = dz_1hr.iloc[-1]

    buy1 = max(dz_1hr_0, dz_1hr_1)
    buy2 = (dz_1hr_0 + dz_1hr_1)/2

    sell1 = min(sz_1hr_0, sz_1hr_1)
    sell2 = (sz_1hr_0 + sz_1hr_1)/2

    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position()

    print(f'pos size is {pos_size} im in pos is {im_in_pos} pnl perc is {pnl_perc}')

    openorderslist = get_open_order_prices()
    openorderslist = [float(d) for d in openorderslist]

    if buy2 and sell2 in openorderslist:
        new_orders_needed = False
        print('buy2 and sell2 in open orders')
    else:
        new_orders_needed = True
        print('no open orders')

    config = utils.get_config()
    account: LocalAccount = eth_account.Account.from_key(config["secret_key"])
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    acct_value = user_state["marginSummary"]["accountValue"]
    acct_value = float(acct_value)
    
    no_trading = get_atr_notrading() #this returns no_trading if atr is over max atr
    if acct_value < min_acct_value:
        no_trading = True

    signal_01 = sma_signal() # 1 is long, 0 is short None is no trading

    if signal_01 == 1:
        long_only = True
        short_only = False
    elif signal_01 == 0:
        short_only = True
        long_only = False
    else:
        no_trading = True
        long_only = False
        short_only = False
    
    print(f'lonmg only:{long_only} short only:{short_only} no_trading:{no_trading}')

    if not im_in_pos and new_orders_needed == True and no_trading == False:

        print('not in pos')

        cancel_all_orders()

        if long_only == True:

            buy2 = limit_order(symbol, True, size, buy2, False)
        
        elif short_only == True:

            sell2 = limit_order(symbol, False, size, sell2, False)
        else:
            buy2 = limit_order(symbol, True, size, buy2, False)
            sell2 = limit_order(symbol, False, size, sell2, False)
    
    elif im_in_pos and no_trading == False:
        print('we are in position... checking PNL loss')
        pnl_close()
    elif no_trading == True:
        print(f'no trading is true because acct value: {acct_value} is less than {min_acct_value}')
        cancel_all_orders()
        kill_switch(pos_sym)
    else:
        print('orders already set... chilling')

bot() 
#schedule bot every 1 hour
schedule.every(10).seconds.do(bot)
#86400 day
while True:
    try:
        schedule.run_pending()
    except:
        print('internet problem.. code failed. sleeping 10')
        time.sleep(10)

