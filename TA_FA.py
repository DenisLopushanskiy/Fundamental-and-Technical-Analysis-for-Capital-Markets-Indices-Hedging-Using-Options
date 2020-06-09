import ta
import pandas as pd
#%%
def EMA_sign(close, short, long):
     '''Gives data with 0 and 1 when EMA gives signal to hedge. 
     :close: Series format with closePrice
     :short: period for fast MA
     :long: period for slow MA
     :return: list with 1 when to buy options
     '''
     emashort = close.ewm(span=short, adjust=False).mean()
     emalong = close.ewm(span=long, adjust=False).mean()
     emaSign = [1 if (x - y) < 0 else 0 for x, y in zip(emashort, emalong)]
     return emaSign

#%%
def AD_sign(high, low, short, long, smoothing):
     '''Gives data with 0 and 1 when EMA gives signal to hedge. 
     :high: Series format with highPrice
     :low: Series format with lowPrice 
     :short: period for fast MA
     :long: period for slow MA
     :return: list with 1 when to buy options
     '''
     awesomeOsc = ta.momentum.AwesomeOscillatorIndicator(high=high, low=low, s=short, len=long, fillna=False)._ao
     awOscSMA = awesomeOsc.rolling(window=smoothing).mean()
     accelDeccel = [1 if (x - y) < 0 else 0 for x, y in zip(awesomeOsc,awOscSMA)]
     return accelDeccel

#%%
def KAMA_sign(close, n, pow1, pow2, SMAsmoothing):
     '''Gives data with 0 and 1 when KAMA short crosses down SMA smoothed KAMA. 
     :close: Series format with closePrice
     :n: period for smoothing 
     :pow1: period for fast MA
     :pow2: period for slow MA
     :SMAsmoothing: period for KAMA smooth
     :return: list with 1 when to buy options
     '''
     aMA = ta.momentum.KAMAIndicator(close=close, n=n, pow1=pow1, pow2=pow2, fillna=False)._kama
     aMASMA = pd.Series(aMA).rolling(window=SMAsmoothing).mean()
     aMASign = [1 if (x - y) < 0 else 0 for x, y in zip(aMA, aMASMA)]
     return aMASign

#%%
def MACD_sign(close, slow, fast, sign):
     '''Gives data with 0 and 1 when MACD gives signal to hedge. 
     :close: Series format with closePrice
     :slow: period for slow MA 
     :fast: period for fast MA
     :sign: period for smoothing
     :return: list with 1 when to buy options
     '''
     mACD = list(ta.trend.MACD(close=close, n_slow=slow, n_fast=fast, n_sign=sign, fillna=False)._macd)
     mACDSign = [1 if x < 0 else 0 for x in mACD]
     return mACDSign

#%%    
def TRIX_sign(close, n):
     '''Gives data with 0 and 1 when TRIX gives signal to hedge. 
     :close: Series format with closePrice
     :n: period for smoothing 
     :return: list with 1 when to buy options
     '''
     tRIX = list(ta.trend.TRIXIndicator(close=close, n=n, fillna=False)._trix)
     tRIXSign = [1 if x < 0 else 0 for x in tRIX]
     return tRIXSign     
     
     
     
     
     
     
     