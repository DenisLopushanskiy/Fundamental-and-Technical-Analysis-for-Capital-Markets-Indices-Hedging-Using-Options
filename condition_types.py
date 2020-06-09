import pandas as pd
#from scipy import stats
import numpy as np
#import math
#import ta
#import matplotlib.pyplot as plt
#import copy
import itertools
#import time

from days_to_exec import dates_tuple, get_condition_type1, get_condition_type2, get_condition_type3,get_condition_type4
from TA_FA import EMA_sign, AD_sign, KAMA_sign, MACD_sign, TRIX_sign
from capital import calculate_strategies, first_portfolios_with_given_options, sharpe_ratios_vector
import best_portfolios_t_1_to_3 
import best_portfolios_t_4 
import best_portfolios_t_5
#%%
def type1_sharpes(countrynum, indFA, spotPriceList, clPrSeries, hgPrSeries, lwPrSeries, R, T, F, sigma, daysOfExpiration, daysToRenewInit):
     '''For types 1-4 of conditions descriptive statistics and parameters are calculated similarly. Uses TA_best_portfolios from module best_portfolios_t_1_to_3 (this is quite hard to undestand from the first sight, but this part barely can be shortened, because for each condition types inputs and logics are different).
     :indFA: list with FA indicator
     :whatToHedgeList: list with B&H part of portfolio (bonds index, equity index)
     :clPrSeries, hgPrSeries, lwPrSeries: used for TA indicators calculation
     :R, T, F, sigma, daysOfExpiration, daysToRenewInit: what is required for options calculation
     :output: lists with summary over best strategy (sharpe ratio, num of best option strategy, best FA threshold) and its TA parameters
     '''
     seqFA = list(np.linspace(np.nanmin(indFA),np.nanmax(indFA),5)[1:-1])
     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(40, 160, 7))
     combEMA = list(itertools.product(seqEMAShort, seqEMALong))
     seqAD = list(np.linspace(5, 85, 9))
     seqKAMA = list(np.linspace(30, 100, 8))
     seqMACD = list(np.linspace(30, 100, 8))
     seqTRIX = list(np.linspace(20, 100, 9))

     bestTAParamsForEachFA = np.empty((6, len(seqFA)))
     summaryTable = np.empty((3, len(seqFA)))
     #sharpeMatrixEMA =  np.empty((len(combEMA), 7))
     print('FA type 1 ')
     for i in range(0, len(seqFA)):
          sharpeMatrixEMA =  np.empty((len(combEMA), 7))
          sharpeMatrixAD =  np.empty((len(seqAD), 7))
          sharpeMatrixKAMA =  np.empty((len(seqKAMA), 7))
          sharpeMatrixMACD =  np.empty((len(seqMACD), 7))
          sharpeMatrixTRIX =  np.empty((len(seqTRIX), 7))
          
          condFARaw = [1 if x>seqFA[i] else (float('nan') if pd.isna(x) else 0) for x in indFA]
          condFAType1 = get_condition_type1(condFARaw, T)
          
          for j in range(0, len(combEMA)):
               eMACond = EMA_sign(clPrSeries, int(combEMA[j][0]), int(combEMA[j][1])) 
               fAT1PlusEMA = [x*y for x, y in zip(condFAType1,eMACond)]
               datesTuple = dates_tuple(fAT1PlusEMA, T, daysOfExpiration, daysToRenewInit)#daysToRenewInitFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
               optionStrategies = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT1Portfolio1 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)
               sRVector = sharpe_ratios_vector(fAT1Portfolio1)
               sharpeMatrixEMA[j,:] = sRVector.transpose()
          
          for j in range(0, len(seqAD)):
               aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(seqAD[j])) 
               fAT1PlusAD = [x*y for x, y in zip(condFAType1,aDCond)]
               datesTuple = dates_tuple(fAT1PlusAD, T, daysOfExpiration, daysToRenewInit)
               optionStrategies2 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT1Portfolio2 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
               sRVector = sharpe_ratios_vector(fAT1Portfolio2)
               sharpeMatrixAD[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqKAMA)):
               kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(seqKAMA[j])) 
               fAT1PlusKAMA = [x*y for x, y in zip(condFAType1,kAMACond)]
               datesTuple = dates_tuple(fAT1PlusKAMA, T, daysOfExpiration, daysToRenewInit)
               optionStrategies3 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT1Portfolio3 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)
               sRVector = sharpe_ratios_vector(fAT1Portfolio3)
               sharpeMatrixKAMA[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqMACD)):
               mACDCond = MACD_sign(clPrSeries, 26, 12, int(seqMACD[j])) 
               fAT1PlusMACD = [x*y for x, y in zip(condFAType1,mACDCond)]
               datesTuple = dates_tuple(fAT1PlusMACD, T, daysOfExpiration, daysToRenewInit)
               optionStrategies4 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT1Portfolio4 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)
               sRVector = sharpe_ratios_vector(fAT1Portfolio4)
               sharpeMatrixMACD[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqTRIX)):
               tRIXCond = TRIX_sign(clPrSeries, int(seqTRIX[j]))
               fAT1PlusTRIX = [x*y for x, y in zip(condFAType1,tRIXCond)]
               datesTuple = dates_tuple(fAT1PlusTRIX, T, daysOfExpiration, daysToRenewInit)
               optionStrategies5 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT1Portfolio5 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)
               sRVector = sharpe_ratios_vector(fAT1Portfolio5)
               sharpeMatrixTRIX[j,:] = sRVector.transpose()
          maxValuesMatrix = np.empty((5, 7))
          maxValuesMatrixIndices = np.empty((5, 7))
          maxValuesMatrix[0,:] = pd.DataFrame(sharpeMatrixEMA).max()
          maxValuesMatrixIndices[0,:] = pd.DataFrame(sharpeMatrixEMA).idxmax(axis=0, skipna=True)
          maxValuesMatrix[1,:] = pd.DataFrame(sharpeMatrixAD).max()
          maxValuesMatrixIndices[1,:] = pd.DataFrame(sharpeMatrixAD).idxmax(axis=0, skipna=True)
          maxValuesMatrix[2,:] = pd.DataFrame(sharpeMatrixKAMA).max()
          maxValuesMatrixIndices[2,:] = pd.DataFrame(sharpeMatrixKAMA).idxmax(axis=0, skipna=True)
          maxValuesMatrix[3,:] = pd.DataFrame(sharpeMatrixMACD).max()
          maxValuesMatrixIndices[3,:] = pd.DataFrame(sharpeMatrixMACD).idxmax(axis=0, skipna=True)
          maxValuesMatrix[4,:] = pd.DataFrame(sharpeMatrixTRIX).max()
          maxValuesMatrixIndices[4,:] = pd.DataFrame(sharpeMatrixTRIX).idxmax(axis=0, skipna=True)
          
          eMAListWithParams = [combEMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[0,:]))]
          aDListWithParams = [seqAD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[1,:]))]
          kAMAListWithParams = [seqKAMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[2,:]))]
          mACDListWithParams = [seqMACD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[3,:]))]
          tRIXListWithParams = [seqTRIX[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[4,:]))]
          
          eMABestCapital = best_portfolios_t_1_to_3.EMA_best_portfolios(eMAListWithParams, clPrSeries, condFAType1, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit)
          aDBestCapital = best_portfolios_t_1_to_3.AD_best_portfolios(aDListWithParams, hgPrSeries, lwPrSeries, condFAType1, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, eMABestCapital)#from the second TA function uses previously made portfolios (fro kama ad portfolio will be used)
          kAMABestCapital = best_portfolios_t_1_to_3.KAMA_best_portfolios(kAMAListWithParams, clPrSeries, condFAType1, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, aDBestCapital)
          mACDBestCapital = best_portfolios_t_1_to_3.MACD_best_portfolios(mACDListWithParams, clPrSeries, condFAType1, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, kAMABestCapital)
          tRIXBestCapital = best_portfolios_t_1_to_3.TRIX_best_portfolios(tRIXListWithParams, clPrSeries, condFAType1, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, mACDBestCapital)#there are the final portfolios, which sharpes are later checked for the best one
          
          bestSharpeRatiosVector = sharpe_ratios_vector(tRIXBestCapital)
          bestStrategySharpeRatio = float(pd.DataFrame(bestSharpeRatiosVector).max())
          bestStrategyNumber = int(pd.DataFrame(bestSharpeRatiosVector).idxmax(axis=0, skipna=True))
          bestStrategyParams = [eMAListWithParams[int(bestStrategyNumber)][0],eMAListWithParams[int(bestStrategyNumber)][1], aDListWithParams[int(bestStrategyNumber)],kAMAListWithParams[int(bestStrategyNumber)],mACDListWithParams[int(bestStrategyNumber)],tRIXListWithParams[int(bestStrategyNumber)]]
          summaryTable[2,i] = seqFA[i]
          summaryTable[1,i] = bestStrategyNumber
          summaryTable[0,i] = bestStrategySharpeRatio
          bestTAParamsForEachFA[:,i] = bestStrategyParams
     
     bestCombinationOrder = int(pd.DataFrame(summaryTable[0, :]).idxmax(axis=0, skipna=True))
     bestSharpeOverall = summaryTable[0,bestCombinationOrder]
     bestStrNumOverall = int(summaryTable[1,bestCombinationOrder])
     bestFAValueOverall = summaryTable[2,bestCombinationOrder]
     statisticsOverall = [bestSharpeOverall, bestStrNumOverall, bestFAValueOverall]
     bestParamsOverall = bestTAParamsForEachFA[:, bestCombinationOrder]
     return statisticsOverall, list(bestParamsOverall)

#%%
def type2_sharpes(countrynum, indFA, spotPriceList, clPrSeries, hgPrSeries, lwPrSeries, R, T, F, sigma, daysOfExpiration, daysToRenewInit):
     '''For types 1-4 of conditions descriptive statistics and parameters are calculated similarly.
     :indFA: list with FA indicator
     :whatToHedgeList: list with B&H part of portfolio (bonds index, equity index)
     :clPrSeries, hgPrSeries, lwPrSeries: used for TA indicators calculation
     :R, T, F, sigma, daysOfExpiration, daysToRenewInit: what is required for options calculation
     :output: lists with summary over best strategy (sharpe ratio, num of best option strategy, best FA threshold) and its TA parameters
     '''
     
     seqFA = list(np.linspace(np.nanmin(indFA),np.nanmax(indFA),5)[1:-1])
     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(40, 160, 7))
     combEMA = list(itertools.product(seqEMAShort, seqEMALong))
     seqAD = list(np.linspace(5, 85, 9))
     seqKAMA = list(np.linspace(30, 100, 8))
     seqMACD = list(np.linspace(30, 100, 8))
     seqTRIX = list(np.linspace(20, 100, 9))

     bestTAParamsForEachFA = np.empty((6, len(seqFA)))
     summaryTable = np.empty((3, len(seqFA)))
     print('FA type 2 ')
     for i in range(0, len(seqFA)):
          sharpeMatrixEMA =  np.empty((len(combEMA), 7))
          sharpeMatrixAD =  np.empty((len(seqAD), 7))
          sharpeMatrixKAMA =  np.empty((len(seqKAMA), 7))
          sharpeMatrixMACD =  np.empty((len(seqMACD), 7))
          sharpeMatrixTRIX =  np.empty((len(seqTRIX), 7))
          
          
          condFARaw = [1 if x>seqFA[i] else (float('nan') if pd.isna(x) else 0) for x in indFA]
          condFAType2 = get_condition_type2(condFARaw)
          
          for j in range(0, len(combEMA)):
               eMACond = EMA_sign(clPrSeries, int(combEMA[j][0]), int(combEMA[j][1])) 
               fAT2PlusEMA = [x*y for x, y in zip(condFAType2,eMACond)]
               datesTuple = dates_tuple(fAT2PlusEMA, T, daysOfExpiration, daysToRenewInit)#daysToRenewInitFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
               optionStrategies = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT2Portfolio1 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)
               sRVector = sharpe_ratios_vector(fAT2Portfolio1)
               sharpeMatrixEMA[j,:] = sRVector.transpose()
          
          for j in range(0, len(seqAD)):
               aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(seqAD[j])) 
               fAT2PlusAD = [x*y for x, y in zip(condFAType2,aDCond)]
               datesTuple = dates_tuple(fAT2PlusAD, T, daysOfExpiration, daysToRenewInit)
               optionStrategies2 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT2Portfolio2 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
               sRVector = sharpe_ratios_vector(fAT2Portfolio2)
               sharpeMatrixAD[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqKAMA)):
               kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(seqKAMA[j])) 
               fAT2PlusKAMA = [x*y for x, y in zip(condFAType2,kAMACond)]
               datesTuple = dates_tuple(fAT2PlusKAMA, T, daysOfExpiration, daysToRenewInit)
               optionStrategies3 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT2Portfolio3 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)
               sRVector = sharpe_ratios_vector(fAT2Portfolio3)
               sharpeMatrixKAMA[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqMACD)):
               mACDCond = MACD_sign(clPrSeries, 26, 12, int(seqMACD[j])) 
               fAT2PlusMACD = [x*y for x, y in zip(condFAType2,mACDCond)]
               datesTuple = dates_tuple(fAT2PlusMACD, T, daysOfExpiration, daysToRenewInit)
               optionStrategies4 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT2Portfolio4 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)
               sRVector = sharpe_ratios_vector(fAT2Portfolio4)
               sharpeMatrixMACD[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqTRIX)):
               tRIXCond = TRIX_sign(clPrSeries, int(seqTRIX[j]))
               fAT2PlusTRIX = [x*y for x, y in zip(condFAType2,tRIXCond)]
               datesTuple = dates_tuple(fAT2PlusTRIX, T, daysOfExpiration, daysToRenewInit)
               optionStrategies5 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT2Portfolio5 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)
               sRVector = sharpe_ratios_vector(fAT2Portfolio5)
               sharpeMatrixTRIX[j,:] = sRVector.transpose()
          maxValuesMatrix = np.empty((5, 7))
          maxValuesMatrixIndices = np.empty((5, 7))
          maxValuesMatrix[0,:] = pd.DataFrame(sharpeMatrixEMA).max()
          maxValuesMatrixIndices[0,:] = pd.DataFrame(sharpeMatrixEMA).idxmax(axis=0, skipna=True)
          maxValuesMatrix[1,:] = pd.DataFrame(sharpeMatrixAD).max()
          maxValuesMatrixIndices[1,:] = pd.DataFrame(sharpeMatrixAD).idxmax(axis=0, skipna=True)
          maxValuesMatrix[2,:] = pd.DataFrame(sharpeMatrixKAMA).max()
          maxValuesMatrixIndices[2,:] = pd.DataFrame(sharpeMatrixKAMA).idxmax(axis=0, skipna=True)
          maxValuesMatrix[3,:] = pd.DataFrame(sharpeMatrixMACD).max()
          maxValuesMatrixIndices[3,:] = pd.DataFrame(sharpeMatrixMACD).idxmax(axis=0, skipna=True)
          maxValuesMatrix[4,:] = pd.DataFrame(sharpeMatrixTRIX).max()
          maxValuesMatrixIndices[4,:] = pd.DataFrame(sharpeMatrixTRIX).idxmax(axis=0, skipna=True)
          
          eMAListWithParams = [combEMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[0,:]))]
          aDListWithParams = [seqAD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[1,:]))]
          kAMAListWithParams = [seqKAMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[2,:]))]
          mACDListWithParams = [seqMACD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[3,:]))]
          tRIXListWithParams = [seqTRIX[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[4,:]))]
          
          eMABestCapital = best_portfolios_t_1_to_3.EMA_best_portfolios(eMAListWithParams, clPrSeries, condFAType2, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit)
          
          aDBestCapital = best_portfolios_t_1_to_3.AD_best_portfolios(aDListWithParams, hgPrSeries, lwPrSeries, condFAType2, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, eMABestCapital)#from the second TA function uses previously made portfolios (fro kama ad portfolio will be used)
          
          kAMABestCapital = best_portfolios_t_1_to_3.KAMA_best_portfolios(kAMAListWithParams, clPrSeries, condFAType2, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, aDBestCapital)
          
          mACDBestCapital = best_portfolios_t_1_to_3.MACD_best_portfolios(mACDListWithParams, clPrSeries, condFAType2, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, kAMABestCapital)
          
          tRIXBestCapital = best_portfolios_t_1_to_3.TRIX_best_portfolios(tRIXListWithParams, clPrSeries, condFAType2, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, mACDBestCapital)#there are the final portfolios, which sharpes are later checked for the best one
          bestSharpeRatiosVector = sharpe_ratios_vector(tRIXBestCapital)
          bestStrategySharpeRatio = float(pd.DataFrame(bestSharpeRatiosVector).max())
          bestStrategyNumber = int(pd.DataFrame(bestSharpeRatiosVector).idxmax(axis=0, skipna=True))
          bestStrategyParams = [eMAListWithParams[int(bestStrategyNumber)][0],eMAListWithParams[int(bestStrategyNumber)][1], aDListWithParams[int(bestStrategyNumber)],kAMAListWithParams[int(bestStrategyNumber)],mACDListWithParams[int(bestStrategyNumber)],tRIXListWithParams[int(bestStrategyNumber)]]
          summaryTable[2,i] = seqFA[i]
          summaryTable[1,i] = bestStrategyNumber
          summaryTable[0,i] = bestStrategySharpeRatio
          bestTAParamsForEachFA[:,i] = bestStrategyParams
     
     bestCombinationOrder = int(pd.DataFrame(summaryTable[0, :]).idxmax(axis=0, skipna=True))
     bestSharpeOverall = summaryTable[0,bestCombinationOrder]
     bestStrNumOverall = int(summaryTable[1,bestCombinationOrder])
     bestFAValueOverall = summaryTable[2,bestCombinationOrder]
     statisticsOverall = [bestSharpeOverall, bestStrNumOverall, bestFAValueOverall]
     bestParamsOverall = bestTAParamsForEachFA[:, bestCombinationOrder]
     return statisticsOverall, list(bestParamsOverall)

#%%
def type3_sharpes(countrynum, indFA, spotPriceList, clPrSeries, hgPrSeries, lwPrSeries, R, T, F, sigma, daysOfExpiration, daysToRenewInit):
     '''For types 1-4 of conditions descriptive statistics and parameters are calculated similarly.
     :indFA: list with FA indicator
     :whatToHedgeList: list with B&H part of portfolio (bonds index, equity index)
     :clPrSeries, hgPrSeries, lwPrSeries: used for TA indicators calculation
     :R, T, F, sigma, daysOfExpiration, daysToRenewInit: what is required for options calculation
     :output: lists with summary over best strategy (sharpe ratio, num of best option strategy, best FA threshold) and its TA parameters
     '''
     seqFA = list(np.linspace(np.nanmin(indFA),np.nanmax(indFA),5)[1:-1])
     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(40, 160, 7))
     combEMA = list(itertools.product(seqEMAShort, seqEMALong))
     seqAD = list(np.linspace(5, 85, 9))
     seqKAMA = list(np.linspace(30, 100, 8))
     seqMACD = list(np.linspace(30, 100, 8))
     seqTRIX = list(np.linspace(20, 100, 9))

     bestTAParamsForEachFA = np.empty((6, len(seqFA)))
     summaryTable = np.empty((3, len(seqFA)))
     print('FA type 3')
     for i in range(0, len(seqFA)):
          sharpeMatrixEMA =  np.empty((len(combEMA), 7))
          sharpeMatrixAD =  np.empty((len(seqAD), 7))
          sharpeMatrixKAMA =  np.empty((len(seqKAMA), 7))
          sharpeMatrixMACD =  np.empty((len(seqMACD), 7))
          sharpeMatrixTRIX =  np.empty((len(seqTRIX), 7))
          
          
          condFARaw = [1 if x>seqFA[i] else (float('nan') if pd.isna(x) else 0) for x in indFA]
          condFAType3 = get_condition_type3(indFA, condFARaw)
          for j in range(0, len(combEMA)):
               eMACond = EMA_sign(clPrSeries, int(combEMA[j][0]), int(combEMA[j][1])) 
               fAT3PlusEMA = [x*y for x, y in zip(condFAType3,eMACond)]
               datesTuple = dates_tuple(fAT3PlusEMA, T, daysOfExpiration, daysToRenewInit)#daysToRenewInitFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
               optionStrategies = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio1 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)
               sRVector = sharpe_ratios_vector(fAT3Portfolio1)
               sharpeMatrixEMA[j,:] = sRVector.transpose()
          
          for j in range(0, len(seqAD)):
               aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(seqAD[j])) 
               fAT3PlusAD = [x*y for x, y in zip(condFAType3,aDCond)]
               datesTuple = dates_tuple(fAT3PlusAD, T, daysOfExpiration, daysToRenewInit)
               optionStrategies2 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio2 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
               sRVector = sharpe_ratios_vector(fAT3Portfolio2)
               sharpeMatrixAD[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqKAMA)):
               kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(seqKAMA[j])) 
               fAT3PlusKAMA = [x*y for x, y in zip(condFAType3,kAMACond)]
               datesTuple = dates_tuple(fAT3PlusKAMA, T, daysOfExpiration, daysToRenewInit)
               optionStrategies3 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio3 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)
               sRVector = sharpe_ratios_vector(fAT3Portfolio3)
               sharpeMatrixKAMA[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqMACD)):
               mACDCond = MACD_sign(clPrSeries, 26, 12, int(seqMACD[j])) 
               fAT3PlusMACD = [x*y for x, y in zip(condFAType3,mACDCond)]
               datesTuple = dates_tuple(fAT3PlusMACD, T, daysOfExpiration, daysToRenewInit)
               optionStrategies4 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio4 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)
               sRVector = sharpe_ratios_vector(fAT3Portfolio4)
               sharpeMatrixMACD[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqTRIX)):
               tRIXCond = TRIX_sign(clPrSeries, int(seqTRIX[j]))
               fAT3PlusTRIX = [x*y for x, y in zip(condFAType3,tRIXCond)]
               datesTuple = dates_tuple(fAT3PlusTRIX, T, daysOfExpiration, daysToRenewInit)
               optionStrategies5 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio5 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)
               sRVector = sharpe_ratios_vector(fAT3Portfolio5)
               sharpeMatrixTRIX[j,:] = sRVector.transpose()
          maxValuesMatrix = np.empty((5, 7))
          maxValuesMatrixIndices = np.empty((5, 7))
          maxValuesMatrix[0,:] = pd.DataFrame(sharpeMatrixEMA).max()
          maxValuesMatrixIndices[0,:] = pd.DataFrame(sharpeMatrixEMA).idxmax(axis=0, skipna=True)
          maxValuesMatrix[1,:] = pd.DataFrame(sharpeMatrixAD).max()
          maxValuesMatrixIndices[1,:] = pd.DataFrame(sharpeMatrixAD).idxmax(axis=0, skipna=True)
          maxValuesMatrix[2,:] = pd.DataFrame(sharpeMatrixKAMA).max()
          maxValuesMatrixIndices[2,:] = pd.DataFrame(sharpeMatrixKAMA).idxmax(axis=0, skipna=True)
          maxValuesMatrix[3,:] = pd.DataFrame(sharpeMatrixMACD).max()
          maxValuesMatrixIndices[3,:] = pd.DataFrame(sharpeMatrixMACD).idxmax(axis=0, skipna=True)
          maxValuesMatrix[4,:] = pd.DataFrame(sharpeMatrixTRIX).max()
          maxValuesMatrixIndices[4,:] = pd.DataFrame(sharpeMatrixTRIX).idxmax(axis=0, skipna=True)
          
          eMAListWithParams = [combEMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[0,:]))]
          aDListWithParams = [seqAD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[1,:]))]
          kAMAListWithParams = [seqKAMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[2,:]))]
          mACDListWithParams = [seqMACD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[3,:]))]
          tRIXListWithParams = [seqTRIX[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[4,:]))]
          
          eMABestCapital = best_portfolios_t_1_to_3.EMA_best_portfolios(eMAListWithParams, clPrSeries, condFAType3, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit)
          
          aDBestCapital = best_portfolios_t_1_to_3.AD_best_portfolios(aDListWithParams, hgPrSeries, lwPrSeries, condFAType3, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, eMABestCapital)#from the second TA function uses previously made portfolios (fro kama ad portfolio will be used)
          
          kAMABestCapital = best_portfolios_t_1_to_3.KAMA_best_portfolios(kAMAListWithParams, clPrSeries, condFAType3, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, aDBestCapital)
          
          mACDBestCapital = best_portfolios_t_1_to_3.MACD_best_portfolios(mACDListWithParams, clPrSeries, condFAType3, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, kAMABestCapital)
          
          tRIXBestCapital = best_portfolios_t_1_to_3.TRIX_best_portfolios(tRIXListWithParams, clPrSeries, condFAType3, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, mACDBestCapital)#there are the final portfolios, which sharpes are later checked for the best one
          bestSharpeRatiosVector = sharpe_ratios_vector(tRIXBestCapital)
          bestStrategySharpeRatio = float(pd.DataFrame(bestSharpeRatiosVector).max())
          bestStrategyNumber = int(pd.DataFrame(bestSharpeRatiosVector).idxmax(axis=0, skipna=True))
          bestStrategyParams = [eMAListWithParams[int(bestStrategyNumber)][0],eMAListWithParams[int(bestStrategyNumber)][1], aDListWithParams[int(bestStrategyNumber)],kAMAListWithParams[int(bestStrategyNumber)],mACDListWithParams[int(bestStrategyNumber)],tRIXListWithParams[int(bestStrategyNumber)]]
          summaryTable[2,i] = seqFA[i]
          summaryTable[1,i] = bestStrategyNumber
          summaryTable[0,i] = bestStrategySharpeRatio
          bestTAParamsForEachFA[:,i] = bestStrategyParams
     
     bestCombinationOrder = int(pd.DataFrame(summaryTable[0, :]).idxmax(axis=0, skipna=True))
     bestSharpeOverall = summaryTable[0,bestCombinationOrder]
     bestStrNumOverall = int(summaryTable[1,bestCombinationOrder])
     bestFAValueOverall = summaryTable[2,bestCombinationOrder]
     statisticsOverall = [bestSharpeOverall, bestStrNumOverall, bestFAValueOverall]
     bestParamsOverall = bestTAParamsForEachFA[:, bestCombinationOrder]
     return statisticsOverall, list(bestParamsOverall)

#%%
def type4_sharpes(countrynum, indFA, spotPriceList, clPrSeries, hgPrSeries, lwPrSeries, R, T, F, sigma, daysOfExpiration, daysToRenewInit):
     '''For types 1-4 of conditions descriptive statistics and parameters are calculated similarly.
     :indFA: list with FA indicator
     :whatToHedgeList: list with B&H part of portfolio (bonds index, equity index)
     :clPrSeries, hgPrSeries, lwPrSeries: used for TA indicators calculation
     :R, T, F, sigma, daysOfExpiration, daysToRenewInit: what is required for options calculation
     :output: lists with summary over best strategy (sharpe ratio, num of best option strategy, best FA threshold) and its TA parameters
     '''
     seqFA = list(np.linspace(np.nanmin(indFA),np.nanmax(indFA),5)[1:-1])
     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(40, 160, 7))
     combEMA = list(itertools.product(seqEMAShort, seqEMALong))
     seqAD = list(np.linspace(5, 85, 9))
     seqKAMA = list(np.linspace(30, 100, 8))
     seqMACD = list(np.linspace(30, 100, 8))
     seqTRIX = list(np.linspace(20, 100, 9))

     bestTAParamsForEachFA = np.empty((6, len(seqFA)))
     summaryTable = np.empty((3, len(seqFA)))
     print('FA type 4')
     for i in range(0, len(seqFA)):
          sharpeMatrixEMA =  np.empty((len(combEMA), 7))
          sharpeMatrixAD =  np.empty((len(seqAD), 7))
          sharpeMatrixKAMA =  np.empty((len(seqKAMA), 7))
          sharpeMatrixMACD =  np.empty((len(seqMACD), 7))
          sharpeMatrixTRIX =  np.empty((len(seqTRIX), 7))
          
          
          condFARaw = [1 if x>seqFA[i] else (float('nan') if pd.isna(x) else 0) for x in indFA]
          for j in range(0, len(combEMA)):
               eMACond = EMA_sign(clPrSeries, int(combEMA[j][0]), int(combEMA[j][1]))
               condFAType4EMA = get_condition_type4(condFARaw, eMACond)
               fAT4PlusEMA = [x*y for x, y in zip(condFAType4EMA, eMACond)]
               datesTuple = dates_tuple(fAT4PlusEMA, T, daysOfExpiration, daysToRenewInit)#daysToRenewInitFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
               optionStrategies = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio1 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)
               sRVector = sharpe_ratios_vector(fAT3Portfolio1)
               sharpeMatrixEMA[j,:] = sRVector.transpose()
          
          for j in range(0, len(seqAD)):
               aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(seqAD[j])) 
               condFAType4AD = get_condition_type4(condFARaw, aDCond)
               fAT4PlusAD = [x*y for x, y in zip(condFAType4AD, aDCond)]
               datesTuple = dates_tuple(fAT4PlusAD, T, daysOfExpiration, daysToRenewInit)
               optionStrategies2 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio2 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
               sRVector = sharpe_ratios_vector(fAT3Portfolio2)
               sharpeMatrixAD[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqKAMA)):
               kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(seqKAMA[j])) 
               condFAType4KAMA = get_condition_type4(condFARaw, kAMACond)
               fAT4PlusKAMA = [x*y for x, y in zip(condFAType4KAMA, kAMACond)]
               datesTuple = dates_tuple(fAT4PlusKAMA, T, daysOfExpiration, daysToRenewInit)
               optionStrategies3 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio3 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)
               sRVector = sharpe_ratios_vector(fAT3Portfolio3)
               sharpeMatrixKAMA[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqMACD)):
               mACDCond = MACD_sign(clPrSeries, 26, 12, int(seqMACD[j])) 
               condFAType4MACD = get_condition_type4(condFARaw, mACDCond)
               fAT4PlusMACD = [x*y for x, y in zip(condFAType4MACD, mACDCond)]
               datesTuple = dates_tuple(fAT4PlusMACD, T, daysOfExpiration, daysToRenewInit)
               optionStrategies4 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio4 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)
               sRVector = sharpe_ratios_vector(fAT3Portfolio4)
               sharpeMatrixMACD[j,:] = sRVector.transpose()
               
          for j in range(0, len(seqTRIX)):
               tRIXCond = TRIX_sign(clPrSeries, int(seqTRIX[j]))
               condFAType4TRIX = get_condition_type4(condFARaw, tRIXCond)
               fAT4PlusTRIX = [x*y for x, y in zip(condFAType4TRIX, tRIXCond)]
               datesTuple = dates_tuple(fAT4PlusTRIX, T, daysOfExpiration, daysToRenewInit)
               optionStrategies5 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
               fAT3Portfolio5 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)
               sRVector = sharpe_ratios_vector(fAT3Portfolio5)
               sharpeMatrixTRIX[j,:] = sRVector.transpose()
          maxValuesMatrix = np.empty((5, 7))
          maxValuesMatrixIndices = np.empty((5, 7))
          maxValuesMatrix[0,:] = pd.DataFrame(sharpeMatrixEMA).max()
          maxValuesMatrixIndices[0,:] = pd.DataFrame(sharpeMatrixEMA).idxmax(axis=0, skipna=True)
          maxValuesMatrix[1,:] = pd.DataFrame(sharpeMatrixAD).max()
          maxValuesMatrixIndices[1,:] = pd.DataFrame(sharpeMatrixAD).idxmax(axis=0, skipna=True)
          maxValuesMatrix[2,:] = pd.DataFrame(sharpeMatrixKAMA).max()
          maxValuesMatrixIndices[2,:] = pd.DataFrame(sharpeMatrixKAMA).idxmax(axis=0, skipna=True)
          maxValuesMatrix[3,:] = pd.DataFrame(sharpeMatrixMACD).max()
          maxValuesMatrixIndices[3,:] = pd.DataFrame(sharpeMatrixMACD).idxmax(axis=0, skipna=True)
          maxValuesMatrix[4,:] = pd.DataFrame(sharpeMatrixTRIX).max()
          maxValuesMatrixIndices[4,:] = pd.DataFrame(sharpeMatrixTRIX).idxmax(axis=0, skipna=True)
          
          eMAListWithParams = [combEMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[0,:]))]
          aDListWithParams = [seqAD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[1,:]))]
          kAMAListWithParams = [seqKAMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[2,:]))]
          mACDListWithParams = [seqMACD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[3,:]))]
          tRIXListWithParams = [seqTRIX[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[4,:]))]
          
          eMABestCapital = best_portfolios_t_4.EMA_best_portfolios(eMAListWithParams, clPrSeries, condFARaw, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit)
          
          aDBestCapital = best_portfolios_t_4.AD_best_portfolios(aDListWithParams, hgPrSeries, lwPrSeries, condFARaw, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, eMABestCapital)#from the second TA function uses previously made portfolios (fro kama ad portfolio will be used)
          
          kAMABestCapital = best_portfolios_t_4.KAMA_best_portfolios(kAMAListWithParams, clPrSeries, condFARaw, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, aDBestCapital)
          
          mACDBestCapital = best_portfolios_t_4.MACD_best_portfolios(mACDListWithParams, clPrSeries, condFARaw, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, kAMABestCapital)
          
          tRIXBestCapital = best_portfolios_t_4.TRIX_best_portfolios(tRIXListWithParams, clPrSeries, condFARaw, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, mACDBestCapital)#there are the final portfolios, which sharpes are later checked for the best one
          bestSharpeRatiosVector = sharpe_ratios_vector(tRIXBestCapital)
          bestStrategySharpeRatio = float(pd.DataFrame(bestSharpeRatiosVector).max())
          bestStrategyNumber = int(pd.DataFrame(bestSharpeRatiosVector).idxmax(axis=0, skipna=True))
          bestStrategyParams = [eMAListWithParams[int(bestStrategyNumber)][0],eMAListWithParams[int(bestStrategyNumber)][1], aDListWithParams[int(bestStrategyNumber)],kAMAListWithParams[int(bestStrategyNumber)],mACDListWithParams[int(bestStrategyNumber)],tRIXListWithParams[int(bestStrategyNumber)]]
          summaryTable[2,i] = seqFA[i]
          summaryTable[1,i] = bestStrategyNumber
          summaryTable[0,i] = bestStrategySharpeRatio
          bestTAParamsForEachFA[:,i] = bestStrategyParams
          
     bestCombinationOrder = int(pd.DataFrame(summaryTable[0, :]).idxmax(axis=0, skipna=True))
     bestSharpeOverall = summaryTable[0,bestCombinationOrder]
     bestStrNumOverall = int(summaryTable[1,bestCombinationOrder])
     bestFAValueOverall = summaryTable[2,bestCombinationOrder]
     statisticsOverall = [bestSharpeOverall, bestStrNumOverall, bestFAValueOverall]
     bestParamsOverall = bestTAParamsForEachFA[:, bestCombinationOrder]
     return statisticsOverall, list(bestParamsOverall)

#%%
def type5_sharpes(countrynum, spotPriceList, clPrSeries, hgPrSeries, lwPrSeries, R, T, F, sigma, daysOfExpiration, daysToRenewInit):
     '''For type 5 condition descriptive statistics and parameters are calculated. No FA indicators are used in calculation
     :whatToHedgeList: list with B&H part of portfolio (bonds index, equity index)
     :clPrSeries, hgPrSeries, lwPrSeries: used for TA indicators calculation
     :R, T, F, sigma, daysOfExpiration, daysToRenewInit: what is required for options calculation
     :output: lists with strategy summary (sharpe ratio, num of best option strategy) and its TA parameters
     '''

     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(40, 160, 7))
     combEMA = list(itertools.product(seqEMAShort, seqEMALong))
     seqAD = list(np.linspace(5, 85, 9))
     seqKAMA = list(np.linspace(30, 100, 8))
     seqMACD = list(np.linspace(30, 100, 8))
     seqTRIX = list(np.linspace(20, 100, 9))

     summaryTable = []
     
     print('FA type 5')
     sharpeMatrixEMA =  np.empty((len(combEMA), 7))
     sharpeMatrixAD =  np.empty((len(seqAD), 7))
     sharpeMatrixKAMA =  np.empty((len(seqKAMA), 7))
     sharpeMatrixMACD =  np.empty((len(seqMACD), 7))
     sharpeMatrixTRIX =  np.empty((len(seqTRIX), 7))
          
     for j in range(0, len(combEMA)):
          eMACond = EMA_sign(clPrSeries, int(combEMA[j][0]), int(combEMA[j][1]))
          datesTuple = dates_tuple(eMACond, T, daysOfExpiration, daysToRenewInit)#daysToRenewInitFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
          optionStrategies = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio1 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)
          sRVector = sharpe_ratios_vector(fAT3Portfolio1)
          sharpeMatrixEMA[j,:] = sRVector.transpose()
          
     for j in range(0, len(seqAD)):
          aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(seqAD[j])) 
          datesTuple = dates_tuple(aDCond, T, daysOfExpiration, daysToRenewInit)
          optionStrategies2 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio2 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
          sRVector = sharpe_ratios_vector(fAT3Portfolio2)
          sharpeMatrixAD[j,:] = sRVector.transpose()
               
     for j in range(0, len(seqKAMA)):
          kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(seqKAMA[j])) 
          datesTuple = dates_tuple(kAMACond, T, daysOfExpiration, daysToRenewInit)
          optionStrategies3 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio3 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)
          sRVector = sharpe_ratios_vector(fAT3Portfolio3)
          sharpeMatrixKAMA[j,:] = sRVector.transpose()
               
     for j in range(0, len(seqMACD)):
          mACDCond = MACD_sign(clPrSeries, 26, 12, int(seqMACD[j])) 
          datesTuple = dates_tuple(mACDCond, T, daysOfExpiration, daysToRenewInit)
          optionStrategies4 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio4 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)
          sRVector = sharpe_ratios_vector(fAT3Portfolio4)
          sharpeMatrixMACD[j,:] = sRVector.transpose()
               
     for j in range(0, len(seqTRIX)):
          tRIXCond = TRIX_sign(clPrSeries, int(seqTRIX[j]))
          datesTuple = dates_tuple(tRIXCond, T, daysOfExpiration, daysToRenewInit)
          optionStrategies5 = calculate_strategies(countrynum, R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio5 = first_portfolios_with_given_options(spotPriceList, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)
          sRVector = sharpe_ratios_vector(fAT3Portfolio5)
          sharpeMatrixTRIX[j,:] = sRVector.transpose()
     
     maxValuesMatrix = np.empty((5, 7))
     maxValuesMatrixIndices = np.empty((5, 7))
     maxValuesMatrix[0,:] = pd.DataFrame(sharpeMatrixEMA).max()
     maxValuesMatrixIndices[0,:] = pd.DataFrame(sharpeMatrixEMA).idxmax(axis=0, skipna=True)
     maxValuesMatrix[1,:] = pd.DataFrame(sharpeMatrixAD).max()
     maxValuesMatrixIndices[1,:] = pd.DataFrame(sharpeMatrixAD).idxmax(axis=0, skipna=True)
     maxValuesMatrix[2,:] = pd.DataFrame(sharpeMatrixKAMA).max()
     maxValuesMatrixIndices[2,:] = pd.DataFrame(sharpeMatrixKAMA).idxmax(axis=0, skipna=True)
     maxValuesMatrix[3,:] = pd.DataFrame(sharpeMatrixMACD).max()
     maxValuesMatrixIndices[3,:] = pd.DataFrame(sharpeMatrixMACD).idxmax(axis=0, skipna=True)
     maxValuesMatrix[4,:] = pd.DataFrame(sharpeMatrixTRIX).max()
     maxValuesMatrixIndices[4,:] = pd.DataFrame(sharpeMatrixTRIX).idxmax(axis=0, skipna=True)
          
     eMAListWithParams = [combEMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[0,:]))]
     aDListWithParams = [seqAD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[1,:]))]
     kAMAListWithParams = [seqKAMA[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[2,:]))]
     mACDListWithParams = [seqMACD[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[3,:]))]
     tRIXListWithParams = [seqTRIX[int(x)] for ind, x in enumerate(list(maxValuesMatrixIndices[4,:]))]
          
     eMABestCapital = best_portfolios_t_5.EMA_best_portfolios(eMAListWithParams, clPrSeries, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit)
          
     aDBestCapital = best_portfolios_t_5.AD_best_portfolios(aDListWithParams, hgPrSeries, lwPrSeries, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, eMABestCapital)#from the second TA function uses previously made portfolios (fro kama ad portfolio will be used)
          
     kAMABestCapital = best_portfolios_t_5.KAMA_best_portfolios(kAMAListWithParams, clPrSeries, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, aDBestCapital)
          
     mACDBestCapital = best_portfolios_t_5.MACD_best_portfolios(mACDListWithParams, clPrSeries, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, kAMABestCapital)
          
     tRIXBestCapital = best_portfolios_t_5.TRIX_best_portfolios(tRIXListWithParams, clPrSeries, countrynum, R, T, F, sigma, T, daysOfExpiration, daysToRenewInit, mACDBestCapital)#there are the final portfolios, which sharpes are later checked for the best one
     bestSharpeRatiosVector = sharpe_ratios_vector(tRIXBestCapital)
     bestStrategySharpeRatio = float(pd.DataFrame(bestSharpeRatiosVector).max())
     bestStrategyNumber = int(pd.DataFrame(bestSharpeRatiosVector).idxmax(axis=0, skipna=True))
     bestStrategyParams = [eMAListWithParams[int(bestStrategyNumber)][0],eMAListWithParams[int(bestStrategyNumber)][1], aDListWithParams[int(bestStrategyNumber)],kAMAListWithParams[int(bestStrategyNumber)],mACDListWithParams[int(bestStrategyNumber)],tRIXListWithParams[int(bestStrategyNumber)]]
     summaryTable.append(bestStrategySharpeRatio)
     summaryTable.append(bestStrategyNumber)
     
     return summaryTable, bestStrategyParams
