import ccxt
import pandas as pd
import math
import dontshare as ds
import time, schedule
import nice_function as n
from datetime import date, datetime, timezone, tzinfo


phemex = ccxt.phemex ({
    'enableRateLimit': True,    
    'apiKey': ds.Px_apikey ,
    'secret': ds.Px_apiSecret
})

def scanner():
    
    timeframe = '5m'
    nums_bars = 289
    
    #get all markets
    phe_markets = phemex.fetch_currencies()
    exchanges = [phemex]
    
    for exchange in exchanges:
        df = pd.DataFrame()
        for ticker in phe_markets:
            
            tickers = ticker[0:6]
            tickers = (f'{tickers}/USDT') #spot markets
            print(f'{tickers} on {exchange}')
            
            try:
                
                bars = exchange.fetch_ohlcv(tickers, timeframe=timeframe, limit=nums_bars)
                new_df = pd.DataFrame (bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                tickerfound = True
                
                #find support and resistance
                support = new_df['low'].min()
                resistance = new_df['high'].max()
                l2h = resistance - support
                avg = (resistance+support)/2
                print(f'support: {support}, high: {resistance}, l2h: {l2h}, avg: {avg}')
                
                order_book = exchange.fetch_order_book(tickers)
                bid = order_book['bids'][0][0]
                ask = order_book['asks'][0][0]
                print(f'for {tickers} the current bid: {bid} ask {ask}')  
                
                # DATA FRAME FOR SIGNALS
                signal_df = pd.DataFrame()
                signal_df['timestamp'] = new_df['timestamp']
                signal_df['symbol'] = tickers
                signal_df['exchange'] = exchange
                signal_df['support'] = support
                signal_df['resistance'] = resistance
                signal_df['bid'] = bid   
                signal_df['ask'] = ask   
                signal_df['breakOUT'] = False   
                signal_df['breakDOWN'] = False         
                
                if bid > resistance:
                    print(f'**** we have break OUT for {tickers}, bid is HIGHER than RESISTANCE')                    
                    breakout = True
                    signal_df['breakOUT'] = breakout
                elif ask < support:
                    print(f'**** we have break DOWN for {tickers}, bid is LOWER than SUPPORT')                    
                    breakdown = True
                    signal_df['breakDOWN'] = breakdown
                else:
                    print(f'no break out or break down for {tickers}')
                    signal_df['breakOUT'] = breakout
                    signal_df['breakDOWN'] = breakdown
                    
            except:
                print(f'there is no symbol for {tickers}')
                ob = 'nothing'
                tickerfound = False
                
                
            signal_df = signal_df[-1:]
            print(signal_df)
            df = pd.concat([df, signal_df])
            print('')
            print('')
            print('------')
                
            # count the number of breakouts or downs
            print(df.breakDOWN.valuecounts())
            print(df.breakOUT.valuecounts())
            print('the number of symbols is:', len(df.index))
            
            # get the symbol count in order to compare it
            symbol_count = len(df.index)
            
            breakout_num= df['breakOUT'].sum()
            print(f'total number of breakouts are TRUE: {breakout_num}')
            
            breakdown_num= df['breakDOWN'].sum()
            print(f'total number of breakouts are TRUE: {breakdown_num}')
            
            #bullish or bearish signals
            if breakdown_num > breakout_num:
                bullish = False
                print(f'there are more breakdowns than breakouts bullish set to {bullish}')
                
            elif breakout_num > breakdown_num:
                
                bullish = True
                print(f'there are more breakout than breakdown bullish set to {bullish}')
                
            else:
                print(f'there are the same amount of breakouts and breakdowns..')
                
            rowsw_true = df.iloc[(df['breakOUT'] == True) | (df['breakDOWN'] == True)]
            print(rowsw_true) 
            # rowsw_true.to_csv('signal.bo.csv', index=False) 
            
            # if 7% or more of the symbol are breaking out or down, put emergery
            perc_07 = symbol_count * .07
            if breakdown_num > perc_07:
                print('**** EMERGENCY CLOSE ALL** 7 percent symbol are breaking down...')
                perc_07_lows_close = True
            elif breakout_num > perc_07:
                print(('**** EMERGENCY CLOSE ALL** 7 percent symbol are breaking down...'))
                perc_07_his_close = True
            else:
                print('.. no anomlies in breakouts or downs')
                
                
scanner()
