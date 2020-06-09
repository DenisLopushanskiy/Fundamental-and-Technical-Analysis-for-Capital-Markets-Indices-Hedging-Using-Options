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
def capital_for_given_parameters_type1(countrynum, bestFA, numOfObserv, threshold, listOfParams, numOfBestStrategy, closePriceList, closePriceSeries, highPriceSeries, lowPriceSeries, interpolatedRates, daysTillExpiration, vixList, daysOfExpiration):
     '''This function calculates capital dynamics based on parameters from training part. Also it gives best train strategy Sharpe ratio as well as best test Sharpe ratio (to check whether best train strategy is also best in the test sample).
     :bestFA: list with FA indicator determined by condition based on train part output
     :numOfObserv: hom much data to use in test part 
     :threshold: parameter when index is considered overbought. It is given from test part
     :listOfParams: periods of TA indicators to use to check
     :numOfBestStrategy: index for the best option strategy from training part (0 is atm put)
     :closePriceList: B&H part of portfolio
     :closePriceSeries, highPriceSeries, lowPriceSeries: data to use for TA signals calculation
     :interpolatedRates, daysTillExpiration, vixList, daysOfExpiration: is as in options calculation part
     :output: three variables with capital and Sharpe ratios
     '''
     futuresPrice = futures_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], list(closePriceSeries)[-numOfObserv:])
     daysToRenewInit = new_futures_dates(daysOfExpiration[-numOfObserv:], futuresPrice)
     condFARaw = [1 if x>0.8 else (float('nan') if pd.isna(x) else 0) for x in bestFA][-numOfObserv:]
     condFA = get_condition_type1(condFARaw, daysTillExpiration[-numOfObserv:])

     eMACond = EMA_sign(closePriceSeries, int(listOfParams[0]), int(listOfParams[1]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
     optionStrategies = calculate_strategies(countrynum, interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio1 = first_portfolios_with_given_options(closePriceList[-numOfObserv:], datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

     aDCond = AD_sign(highPriceSeries, lowPriceSeries, 5, 34, int(listOfParams[2]))[-numOfObserv:] 
     fAPlusTA = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies2 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio2 = portfolios_with_given_options(fAPortfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
     
     kAMACond = KAMA_sign(closePriceSeries, 10, 2, 30, int(listOfParams[3]))[-numOfObserv:] 
     fAPlusTA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies3 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio3 = portfolios_with_given_options(fAPortfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

     mACDCond = MACD_sign(closePriceSeries, 26, 12, int(listOfParams[4]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies4 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio4 = portfolios_with_given_options(fAPortfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

     tRIXCond = TRIX_sign(closePriceSeries, int(listOfParams[5]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies5 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio5 = portfolios_with_given_options(fAPortfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

     fASharpeRatiosVector = sharpe_ratios_vector(fAPortfolio5)
     bestTrainStrategyCapital = list(fAPortfolio5.iloc[:,numOfBestStrategy])
     bestTrainStrategySharpe = float(fASharpeRatiosVector.iloc[numOfBestStrategy,0])
     bestTestSharpe = float(fASharpeRatiosVector.max())

     return bestTrainStrategyCapital, bestTrainStrategySharpe, bestTestSharpe

#%%
def capital_for_given_parameters_type2(countrynum, bestFA, numOfObserv, threshold, listOfParams, numOfBestStrategy, closePriceList, closePriceSeries, highPriceSeries, lowPriceSeries, interpolatedRates, daysTillExpiration, vixList, daysOfExpiration):
     '''This function calculates capital dynamics based on parameters from training part. Also it gives best train strategy Sharpe ratio as well as best test Sharpe ratio (to check whether best train strategy is also best in the test sample).
     :bestFA: list with FA indicator determined by condition based on train part output
     :numOfObserv: hom much data to use in test part 
     :threshold: parameter when index is considered overbought. It is given from test part
     :listOfParams: periods of TA indicators to use to check
     :numOfBestStrategy: index for the best option strategy from training part (0 is atm put)
     :closePriceList: B&H part of portfolio
     :closePriceSeries, highPriceSeries, lowPriceSeries: data to use for TA signals calculation
     :interpolatedRates, daysTillExpiration, vixList, daysOfExpiration: is as in options calculation part
     :output: three variables with capital and Sharpe ratios
     '''
     futuresPrice = futures_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], list(closePriceSeries)[-numOfObserv:])
     daysToRenewInit = new_futures_dates(daysOfExpiration[-numOfObserv:], futuresPrice)
     condFARaw = [1 if x>threshold else (float('nan') if pd.isna(x) else 0) for x in bestFA][-numOfObserv:]
     condFA = get_condition_type2(condFARaw)
     
     eMACond = EMA_sign(closePriceSeries, int(listOfParams[0]), int(listOfParams[1]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
     optionStrategies = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio1 = first_portfolios_with_given_options(closePriceList[-numOfObserv:], datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

     aDCond = AD_sign(highPriceSeries, lowPriceSeries, 5, 34, int(listOfParams[2]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies2 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio2 = portfolios_with_given_options(fAPortfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
     
     kAMACond = KAMA_sign(closePriceSeries, 10, 2, 30, int(listOfParams[3]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies3 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio3 = portfolios_with_given_options(fAPortfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

     mACDCond = MACD_sign(closePriceSeries, 26, 12, int(listOfParams[4]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies4 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio4 = portfolios_with_given_options(fAPortfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

     tRIXCond = TRIX_sign(closePriceSeries, int(listOfParams[5]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies5 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio5 = portfolios_with_given_options(fAPortfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

     fASharpeRatiosVector = sharpe_ratios_vector(fAPortfolio5)
     bestTrainStrategyCapital = list(fAPortfolio5.iloc[:,numOfBestStrategy])
     bestTrainStrategySharpe = float(fASharpeRatiosVector.iloc[numOfBestStrategy,0])
     bestTestSharpe = float(fASharpeRatiosVector.max())

     return bestTrainStrategyCapital, bestTrainStrategySharpe, bestTestSharpe

#%%
def capital_for_given_parameters_type3(countrynum, bestFA, numOfObserv, threshold, listOfParams, numOfBestStrategy, closePriceList, closePriceSeries, highPriceSeries, lowPriceSeries, interpolatedRates, daysTillExpiration, vixList, daysOfExpiration):
     '''This function calculates capital dynamics based on parameters from training part. Also it gives best train strategy Sharpe ratio as well as best test Sharpe ratio (to check whether best train strategy is also best in the test sample).
     :bestFA: list with FA indicator determined by condition based on train part output
     :numOfObserv: hom much data to use in test part 
     :threshold: parameter when index is considered overbought. It is given from test part
     :listOfParams: periods of TA indicators to use to check
     :numOfBestStrategy: index for the best option strategy from training part (0 is atm put)
     :closePriceList: B&H part of portfolio
     :closePriceSeries, highPriceSeries, lowPriceSeries: data to use for TA signals calculation
     :interpolatedRates, daysTillExpiration, vixList, daysOfExpiration: is as in options calculation part
     :output: three variables with capital and Sharpe ratios
     '''   
     futuresPrice = futures_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], list(closePriceSeries)[-numOfObserv:])
     daysToRenewInit = new_futures_dates(daysOfExpiration[-numOfObserv:], futuresPrice)
     condFARaw = [1 if x>threshold else (float('nan') if pd.isna(x) else 0) for x in bestFA][-numOfObserv:]
     condFA = get_condition_type3(bestFA, condFARaw)
     
     eMACond = EMA_sign(closePriceSeries, int(listOfParams[0]), int(listOfParams[1]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
     optionStrategies = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio1 = first_portfolios_with_given_options(closePriceList[-numOfObserv:], datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

     aDCond = AD_sign(highPriceSeries, lowPriceSeries, 5, 34, int(listOfParams[2]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies2 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio2 = portfolios_with_given_options(fAPortfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
     
     kAMACond = KAMA_sign(closePriceSeries, 10, 2, 30, int(listOfParams[3]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies3 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio3 = portfolios_with_given_options(fAPortfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

     mACDCond = MACD_sign(closePriceSeries, 26, 12, int(listOfParams[4]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies4 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio4 = portfolios_with_given_options(fAPortfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

     tRIXCond = TRIX_sign(closePriceSeries, int(listOfParams[5]))[-numOfObserv:]
     fAPlusTA = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies5 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio5 = portfolios_with_given_options(fAPortfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

     fASharpeRatiosVector = sharpe_ratios_vector(fAPortfolio5)
     bestTrainStrategyCapital = list(fAPortfolio5.iloc[:,numOfBestStrategy])
     bestTrainStrategySharpe = float(fASharpeRatiosVector.iloc[numOfBestStrategy,0])
     bestTestSharpe = float(fASharpeRatiosVector.max())

     return bestTrainStrategyCapital, bestTrainStrategySharpe, bestTestSharpe

#%%
def capital_for_given_parameters_type4(countrynum, bestFA, numOfObserv, threshold, listOfParams, numOfBestStrategy, closePriceList, closePriceSeries, highPriceSeries, lowPriceSeries, interpolatedRates, daysTillExpiration, vixList, daysOfExpiration):
     '''This function calculates capital dynamics based on parameters from training part. Also it gives best train strategy Sharpe ratio as well as best test Sharpe ratio (to check whether best train strategy is also best in the test sample).
     :bestFA: list with FA indicator determined by condition based on train part output
     :numOfObserv: hom much data to use in test part 
     :threshold: parameter when index is considered overbought. It is given from test part
     :listOfParams: periods of TA indicators to use to check
     :numOfBestStrategy: index for the best option strategy from training part (0 is atm put)
     :closePriceList: B&H part of portfolio
     :closePriceSeries, highPriceSeries, lowPriceSeries: data to use for TA signals calculation
     :interpolatedRates, daysTillExpiration, vixList, daysOfExpiration: is as in options calculation part
     :output: three variables with capital and Sharpe ratios
     ''' 
     futuresPrice = futures_price(interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], list(closePriceSeries)[-numOfObserv:])
     daysToRenewInit = new_futures_dates(daysOfExpiration[-numOfObserv:], futuresPrice)
     condFARaw = [1 if x>threshold else (float('nan') if pd.isna(x) else 0) for x in bestFA][-numOfObserv:]
     
     eMACond = EMA_sign(closePriceSeries, int(listOfParams[0]), int(listOfParams[1]))[-numOfObserv:]
     fAPlusEMA = get_condition_type4(condFARaw, eMACond)
     fAPlusTA = [x*y for x, y in zip(fAPlusEMA,eMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
     optionStrategies = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio1 = first_portfolios_with_given_options(closePriceList[-numOfObserv:], datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

     aDCond = AD_sign(highPriceSeries, lowPriceSeries, 5, 34, int(listOfParams[2]))[-numOfObserv:]
     fAPlusAD = get_condition_type4(condFARaw, aDCond)
     fAPlusTA = [x*y for x, y in zip(fAPlusAD,aDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies2 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio2 = portfolios_with_given_options(fAPortfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
     
     kAMACond = KAMA_sign(closePriceSeries, 10, 2, 30, int(listOfParams[3]))[-numOfObserv:]
     fAPlusKAMA = get_condition_type4(condFARaw, kAMACond)
     fAPlusTA = [x*y for x, y in zip(fAPlusKAMA,kAMACond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies3 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio3 = portfolios_with_given_options(fAPortfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

     mACDCond = MACD_sign(closePriceSeries, 26, 12, int(listOfParams[4]))[-numOfObserv:]
     fAPlusMACD = get_condition_type4(condFARaw, mACDCond)
     fAPlusTA = [x*y for x, y in zip(fAPlusMACD,mACDCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies4 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio4 = portfolios_with_given_options(fAPortfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

     tRIXCond = TRIX_sign(closePriceSeries, int(listOfParams[5]))[-numOfObserv:]
     fAPlusTRIX = get_condition_type4(condFARaw, tRIXCond)
     fAPlusTA = [x*y for x, y in zip(fAPlusTRIX,tRIXCond)]
     datesTuple = dates_tuple(fAPlusTA, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies5 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio5 = portfolios_with_given_options(fAPortfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

     fASharpeRatiosVector = sharpe_ratios_vector(fAPortfolio5)
     bestTrainStrategyCapital = list(fAPortfolio5.iloc[:,numOfBestStrategy])
     bestTrainStrategySharpe = float(fASharpeRatiosVector.iloc[numOfBestStrategy,0])
     bestTestSharpe = float(fASharpeRatiosVector.max())

     return bestTrainStrategyCapital, bestTrainStrategySharpe, bestTestSharpe

#%%
def capital_for_given_parameters_type5(countrynum, numOfObserv, listOfParams, numOfBestStrategy, closePriceList, closePriceSeries, highPriceSeries, lowPriceSeries, interpolatedRates, daysTillExpiration, vixList, daysOfExpiration):
     '''This function calculates capital dynamics based on parameters from training part. Also it gives best train strategy Sharpe ratio as well as best test Sharpe ratio (to check whether best train strategy is also best in the test sample).
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
     optionStrategies = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio1 = first_portfolios_with_given_options(closePriceList[-numOfObserv:], datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

     aDCond = AD_sign(highPriceSeries, lowPriceSeries, 5, 34, int(listOfParams[2]))[-numOfObserv:]
     datesTuple = dates_tuple(aDCond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies2 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio2 = portfolios_with_given_options(fAPortfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
     
     kAMACond = KAMA_sign(closePriceSeries, 10, 2, 30, int(listOfParams[3]))[-numOfObserv:]
     datesTuple = dates_tuple(kAMACond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies3 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio3 = portfolios_with_given_options(fAPortfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

     mACDCond = MACD_sign(closePriceSeries, 26, 12, int(listOfParams[4]))[-numOfObserv:]
     datesTuple = dates_tuple(mACDCond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies4 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio4 = portfolios_with_given_options(fAPortfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

     tRIXCond = TRIX_sign(closePriceSeries, int(listOfParams[5]))[-numOfObserv:]
     datesTuple = dates_tuple(tRIXCond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
     optionStrategies5 = calculate_strategies(countrynum,interpolatedRates[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
     fAPortfolio5 = portfolios_with_given_options(fAPortfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

     fASharpeRatiosVector = sharpe_ratios_vector(fAPortfolio5)
     bestTrainStrategyCapital = list(fAPortfolio5.iloc[:,numOfBestStrategy])
     bestTrainStrategySharpe = float(fASharpeRatiosVector.iloc[numOfBestStrategy,0])
     bestTestSharpe = float(fASharpeRatiosVector.max())

     return bestTrainStrategyCapital, bestTrainStrategySharpe, bestTestSharpe

