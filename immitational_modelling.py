import pandas as pd
#from scipy import stats
import numpy as np
import math
#import ta
#import matplotlib.pyplot as plt
import copy
#import itertools
import time

from days_to_exec import days_left, new_futures_dates, give_first_one, give_first_zero, period_to_hold_options, dates_tuple, get_condition_type1, get_condition_type2, get_condition_type3,get_condition_type4
from rates_interpolation import interpolate_rates
from derivatives import futures_price, Option, CallOpt, PutOpt
from TA_FA import EMA_sign, AD_sign, KAMA_sign, MACD_sign, TRIX_sign
from capital import create_portfolio, sharpe_ratio, calculate_strategies, first_portfolios_with_given_options, portfolios_with_given_options, sharpe_ratios_vector
#import best_portfolios_t_1_to_3 
#import best_portfolios_t_4 
#import best_portfolios_t_5
from condition_types import type1_sharpes, type2_sharpes, type3_sharpes, type4_sharpes, type5_sharpes

#%%
def immit_mod_FA_Sharpes(countrynum, bestFA, numOfObserv, threshold, listOfParams, closePriceList, closePriceSeries, highPriceSeries, lowPriceSeries, interpolatedRates, daysTillExpiration, vixList, daysOfExpiration):
     '''This function calculates capital dynamics based on parameters for test part. Output is sharpe ratio for a given strategy type (in USA it is PET2) and best option strategy (putATM - strategy number 0). Used for Monte-Carlo analysis.
     :bestFA: list with FA indicator determined by condition based on train part output
     :numOfObserv: hom much data to use in test part 
     :threshold: parameter when index is considered overbought. It is given from test part
     :listOfParams: periods of TA indicators to use to check
     :closePriceList: B&H part of portfolio
     :closePriceSeries, highPriceSeries, lowPriceSeries: data to use for TA signals calculation
     :interpolatedRates, daysTillExpiration, vixList, daysOfExpiration: is as in options calculation part
     :output: three variables with capital and Sharpe ratios
     '''
     futuresPrice = futures_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], list(closePriceSeries)[-numOfObserv:])
     daysToRenewInit = new_futures_dates(daysOfExpiration[-numOfObserv:], futuresPrice)
     condFARaw = [1 if x>threshold else (float('nan') if pd.isna(x) else 0) for x in bestFA][-numOfObserv:]
     condFA = get_condition_type2(condFARaw)
     #condFA = get_condition_type1(condFARaw, daysTillExpiration[-numOfObserv:])
     #condFA = get_condition_type3(bestFA[-numOfObserv:], condFARaw) #look at the condition the chosen counry has
     
     eMACond = EMA_sign(closePriceSeries, int(listOfParams[0]), int(listOfParams[1]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
     putATM = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     portfolio1 = create_portfolio(closePriceList[-numOfObserv:], datesTuple[2], datesTuple[0], datesTuple[1], putATM)

     aDCond = AD_sign(highPriceSeries, lowPriceSeries, 5, 34, int(listOfParams[2]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     putATM2 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     portfolio2 = create_portfolio(portfolio1, datesTuple[2], datesTuple[0], datesTuple[1], putATM2)
     
     kAMACond = KAMA_sign(closePriceSeries, 10, 2, 30, int(listOfParams[3]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     putATM3 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     portfolio3 = create_portfolio(portfolio2, datesTuple[2], datesTuple[0], datesTuple[1], putATM3)

     mACDCond = MACD_sign(closePriceSeries, 26, 12, int(listOfParams[4]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     putATM4 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     portfolio4 = create_portfolio(portfolio3, datesTuple[2], datesTuple[0], datesTuple[1], putATM4)

     tRIXCond = TRIX_sign(closePriceSeries, int(listOfParams[5]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     putATM5 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     portfolio5 = create_portfolio(portfolio4, datesTuple[2], datesTuple[0], datesTuple[1], putATM5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

     sharpeRatio = sharpe_ratio(portfolio5)
     return sharpeRatio

#%%
def immit_mod_TA_Sharpes(countrynum, numOfObserv, listOfParams, closePriceList, closePriceSeries, highPriceSeries, lowPriceSeries, interpolatedRates, daysTillExpiration, vixList, daysOfExpiration):
     '''This function calculates capital dynamics based on parameters for test part. Output is sharpe ratio for a given best option strategy (strangle10 - strategy number 4). Used for Monte-Carlo analysis.
     :No FA is required as the system is based purely on TA indicators:
     :numOfObserv: hom much data to use in test part 
     :listOfParams: periods of TA indicators to use to check
     :numOfBestStrategy: index for the best option strategy from training part (0 is atm put)
     :closePriceList: B&H part of portfolio
     :closePriceSeries, highPriceSeries, lowPriceSeries: data to use for TA signals calculation
     :interpolatedRates, daysTillExpiration, vixList, daysOfExpiration: is as in options calculation part
     :output: three variables with capital and Sharpe ratios
     '''  
     futuresPrice = futures_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], list(closePriceSeries)[-numOfObserv:])
     daysToRenewInit = new_futures_dates(daysOfExpiration[-numOfObserv:], futuresPrice)
     
     eMACond = EMA_sign(closePriceSeries, int(listOfParams[0]), int(listOfParams[1]))[-numOfObserv:]
     datesTuple = dates_tuple(eMACond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
     #putATM = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     strangle10 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3]))]
     portfolio1 = create_portfolio(closePriceList[-numOfObserv:], datesTuple[2], datesTuple[0], datesTuple[1], strangle10)

     aDCond = AD_sign(highPriceSeries, lowPriceSeries, 5, 34, int(listOfParams[2]))[-numOfObserv:]
     datesTuple = dates_tuple(aDCond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     #putATM2 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     strangle102 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3]))]
     portfolio2 = create_portfolio(portfolio1, datesTuple[2], datesTuple[0], datesTuple[1], strangle102)
     
     kAMACond = KAMA_sign(closePriceSeries, 10, 2, 30, int(listOfParams[3]))[-numOfObserv:]
     datesTuple = dates_tuple(kAMACond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     #putATM3 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     strangle103 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3]))]
     portfolio3 = create_portfolio(portfolio2, datesTuple[2], datesTuple[0], datesTuple[1], strangle103)

     mACDCond = MACD_sign(closePriceSeries, 26, 12, int(listOfParams[4]))[-numOfObserv:]
     datesTuple = dates_tuple(mACDCond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     #putATM4 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     strangle104 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3]))]
     portfolio4 = create_portfolio(portfolio3, datesTuple[2], datesTuple[0], datesTuple[1], strangle104)

     tRIXCond = TRIX_sign(closePriceSeries, int(listOfParams[5]))[-numOfObserv:]
     datesTuple = dates_tuple(tRIXCond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     #putATM5 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])
     strangle105 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3]))]
     portfolio5 = create_portfolio(portfolio4, datesTuple[2], datesTuple[0], datesTuple[1], strangle105)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

     sharpeRatio = sharpe_ratio(portfolio5)

     return sharpeRatio
