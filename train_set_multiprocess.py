import pandas as pd
#from scipy import stats
import numpy as np
#import math
#import ta
#import matplotlib.pyplot as plt
import copy
#import itertools
import time
import multiprocessing  
from concurrent.futures import ProcessPoolExecutor

from days_to_exec import days_left, new_futures_dates, give_first_one, give_first_zero, period_to_hold_options, dates_tuple, get_condition_type1, get_condition_type2, get_condition_type3,get_condition_type4
from rates_interpolation import interpolate_rates
from derivatives import futures_price, Option, CallOpt, PutOpt
from TA_FA import EMA_sign, AD_sign, KAMA_sign, MACD_sign, TRIX_sign
from capital import create_portfolio, sharpe_ratio, calculate_strategies, first_portfolios_with_given_options, portfolios_with_given_options, sharpe_ratios_vector
#import best_portfolios_t_1_to_3 
#import best_portfolios_t_4 
#import best_portfolios_t_5
from condition_types import type1_sharpes, type2_sharpes, type3_sharpes, type4_sharpes, type5_sharpes
from test_portfolios_all_types import capital_for_given_parameters_type1, capital_for_given_parameters_type2, capital_for_given_parameters_type3, capital_for_given_parameters_type4, capital_for_given_parameters_type5


#%% data download
openPriceDF = pd.read_csv('./data/Open_price.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')
highPriceDF = pd.read_csv('./data/High_price.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')
lowPriceDF = pd.read_csv('./data/Low_price.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')
closePriceDF = pd.read_csv('./data/Close_price.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')

rates1W = pd.read_csv('./data/1W.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')
rates1M = pd.read_csv('./data/1M.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')
rates3M = pd.read_csv('./data/3M.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')

indPB = pd.read_csv('./data/PB.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')
indPE = pd.read_csv('./data/PE.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')
indPS = pd.read_csv('./data/PS.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')

vix = pd.read_csv('./data/VIX.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',')

daysLeft = days_left(rates1W)#this is data sensitive, should be left unchanged: this is a tuple with lists inside (0-days to exec, 1-execution dates)

'''
interpolatedRates = np.empty((rates1W.shape[0],rates1W.shape[1]))
for i in range(0,interpolatedRates.shape[1]):
     interpolatedRates[:,i] = interpolate_rates(i, rates1W, rates1M, rates3M, daysLeft[0]) #interpolated rates for each country
interpolatedRates = pd.DataFrame(interpolatedRates)
interpolatedRates.index = closePriceDF.index
interpolatedRates.to_csv('./data/Interpolated_rates.csv', sep=';', decimal=',')
'''
interpolatedRates = pd.read_csv('./data/Interpolated_rates.csv', sep=';', parse_dates=['Dates'], dayfirst=True, index_col='Dates', decimal=',') #added new file to python, but commented code above creates this data set (try to save enough time)


#%%determine length of train and test dataset for each country
firstObservIndices = np.empty((20,6))
for i in range(0,closePriceDF.shape[1]):
     dataSet = list(zip(list(closePriceDF.iloc[:,i]), list(interpolatedRates.iloc[:,i]), list(indPB.iloc[:,i]), list(indPE.iloc[:,i]), list(indPS.iloc[:,i]), list(vix.iloc[:,i])))
     dataFrame = pd.DataFrame(dataSet, columns = ['Close', 'Rates', 'PB', 'PE', 'PS', 'VIX'])
     for j in range(0, firstObservIndices.shape[1]):
          firstObservIndices[i,j] = dataFrame.iloc[:,j].notna().idxmax()
firstObservIndices = pd.DataFrame(firstObservIndices, columns = ['Close', 'Rates', 'PB', 'PE', 'PS', 'VIX'])
spotPriceObservationsNumber = [int(len(closePriceDF) - x) for x in list(firstObservIndices.iloc[:,0])]
minimumObservationsNumber = []#list with number of minimum observations for each country
for i in range(0, firstObservIndices.shape[0]):
     minimumObservationsNumber.append(int(len(closePriceDF) - max(firstObservIndices.iloc[i,:])))
testSetNumOfObserv = [int(min(round(0.3*x, 0),round(0.5*y, 0))) for x, y in zip(spotPriceObservationsNumber, minimumObservationsNumber)] #length of test set
trainSetNumOfObserv = [int(len(closePriceDF) - x) for x in testSetNumOfObserv] #length of training set

#%%determining main train function
def multi_processing(countrynum, pBook, pEarn, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit):
     n_cpus = multiprocessing.cpu_count()
     pool = ProcessPoolExecutor(max_workers=n_cpus)
     futureVector = [float('nan')]*13 #vector where Future objects are stored. The last element contains information about TA system
     futureVector[0] = pool.submit(type1_sharpes, countrynum, pBook, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     futureVector[1] = pool.submit(type1_sharpes, countrynum, pEarn, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     futureVector[2] = pool.submit(type1_sharpes, countrynum, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     
     futureVector[3] = pool.submit(type2_sharpes, countrynum, pBook, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     futureVector[4] = pool.submit(type2_sharpes, countrynum, pEarn, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     futureVector[5] = pool.submit(type2_sharpes, countrynum, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     
     futureVector[6] = pool.submit(type3_sharpes, countrynum, pBook, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     futureVector[7] = pool.submit(type3_sharpes, countrynum, pEarn, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     futureVector[8] = pool.submit(type3_sharpes, countrynum, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     
     futureVector[9] = pool.submit(type4_sharpes, countrynum, pBook, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     futureVector[10] = pool.submit(type4_sharpes, countrynum, pEarn, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     futureVector[11] = pool.submit(type4_sharpes, countrynum, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     
     futureVector[12] = pool.submit(type5_sharpes, countrynum, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)

     return futureVector 
     
#%% TRAINING FOR EACH COUNTRY
buyHoldSharpes = [float('nan')]*closePriceDF.shape[1]#all the Sharpe ratios will be stored here
allCountriesSummary = [float('nan')]*closePriceDF.shape[1]
allCountriesParameters = [float('nan')]*closePriceDF.shape[1]#best parameters for the FA indicators will be stored here
allCountriesTASummary = [float('nan')]*closePriceDF.shape[1]#best option strategy number and best sharpe ratios for each country
allCountriesTAParameters = [float('nan')]*closePriceDF.shape[1]#best TA parameters with system based only on TA indicators shown
#start_time = time.time()

#%%big train cycle
if __name__ == '__main__':
     for countrynum in range(0, closePriceDF.shape[1]):
          print('Training part for {}'.format(closePriceDF.columns[countrynum]))
          start = time.time()
          trainLength = trainSetNumOfObserv[countrynum] 
          daysTillExpirationTr = copy.copy(daysLeft[0][:trainLength])
          daysOfExpirationTr = copy.copy(daysLeft[1][:trainLength])

          interpolatedRatesList = list(interpolatedRates.iloc[:trainLength,countrynum])
          spotPrice = list(closePriceDF.iloc[:trainLength,countrynum])
          clPrSeries = closePriceDF.iloc[:trainLength,countrynum]
          hgPrSeries = highPriceDF.iloc[:trainLength,countrynum]
          lwPrSeries = lowPriceDF.iloc[:trainLength,countrynum]
          
          futuresPrice = futures_price(interpolatedRatesList, daysTillExpirationTr, spotPrice) #list of futures prices for the index in a cycle
          daysToRenewInit = new_futures_dates(daysOfExpirationTr, futuresPrice) 
          vixList = list(vix.iloc[:trainLength,countrynum])

          pBook = list(indPB.iloc[:trainLength,countrynum])
          pEarn = list(indPE.iloc[:trainLength,countrynum])
          pSale = list(indPS.iloc[:trainLength,countrynum])
          
          futureVector = multi_processing(countrynum, pBook, pEarn, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
          
          fASummary = [x.result()[0] for x in futureVector[:-1]]
          fAParameters = [x.result()[1] for x in futureVector[:-1]]
          tASummary = futureVector[-1].result()[0]
          tAParameters = futureVector[-1].result()[1]
          
          summaryMatrix = np.empty((3, 12))#12 strategies with 3 values in output
          parametersMatrix = np.empty((6, 12)) 
          
          for i in range(0,summaryMatrix.shape[1]):
               summaryMatrix[:,i] = fASummary[i]
               parametersMatrix[:,i] = fAParameters[i]
          
          fAStrategyList = ['PBT1', 'PET1', 'PST1', 'PBT2', 'PET2', 'PST2', 'PBT3', 'PET3', 'PST3', 'PBT4', 'PET4', 'PST4']
          
          strategiesAndSummary = pd.DataFrame(summaryMatrix, columns=fAStrategyList)
          strategiesAndParameters = pd.DataFrame(parametersMatrix, columns=fAStrategyList)

          bestStrategySummary = strategiesAndSummary.iloc[:, pd.Series(list(strategiesAndSummary.loc[0,:])).idxmax(axis=0, skipna=True)]
          bestStrategyList = list(bestStrategySummary)
          bestStrategyList.append(list(strategiesAndSummary.columns)[pd.Series(list(strategiesAndSummary.loc[0,:])).idxmax(axis=0, skipna=True)])
          bestParametersList = list(strategiesAndParameters.iloc[:, pd.Series(list(strategiesAndSummary.loc[0,:])).idxmax(axis=0, skipna=True)])

          buyHoldSharpes[countrynum] = sharpe_ratio(spotPrice)
          allCountriesSummary[countrynum] = bestStrategyList
          allCountriesParameters[countrynum] = bestParametersList
          #for j in range(0, 20):
               #allCountriesSummary[j] = [0.45838760695532565, 0.0, 1.5078, 'PST3']
               #allCountriesParameters[j] = [20.0, 30.0, 5.0, 50.0, 30.0, 15.0]
               #allCountriesTASummary[j] = [0.14, 0]
               #allCountriesTAParameters[j] = [14.0, 20.0, 5.0, 40.0, 30.0, 15.0]
          
          #allCountriesSummary[15] = allCountriesParameters[15] = allCountriesTASummary[15] = allCountriesTAParameters[15] = float('nan')
          
          allCountriesTASummary[countrynum] = tASummary
          allCountriesTAParameters[countrynum] = tAParameters
          
          end = time.time()
          print(f'\nOverall time to calculate: {(end - start)/60:.2f}m\n')
          
     #allCountriesSummary = allCountriesSummary[15:]
     #allCountriesParameters = allCountriesParameters[15:]
     #allCountriesTASummary = allCountriesTASummary[15:]
     #allCountriesTAParameters = allCountriesTAParameters[15:]
          
     #print(allCountriesSummary, allCountriesParameters, allCountriesTASummary, allCountriesTAParameters, buyHoldSharpes)
#crucial are: bestStrategyList, bestParametersList - which go to allCountriesSummary and allCountriesParameters, allCountriesTASummary, allCountriesTAParameters

#%%create data frames
     bestFAStrategySummary = pd.DataFrame(allCountriesSummary, index=closePriceDF.columns, columns=['SharpeRatio', 'BestOptionStrategyNum', 'FAThreshold', 'FATypeNum'])
     bestFAStrategyParameters = pd.DataFrame(allCountriesParameters, index=closePriceDF.columns, columns=['EMAshort','EMAlong','AD','KAMA','MACD','TRIX'])
     bestTAStrategySummary = pd.DataFrame(allCountriesTASummary, index=closePriceDF.columns, columns=['SharpeRatio', 'BestOptionStrategyNum'])
     bestTAStrategyParameters = pd.DataFrame(allCountriesTAParameters, index=closePriceDF.columns, columns=['EMAshort','EMAlong','AD','KAMA','MACD','TRIX'])
     bHTrainSharpes = pd.DataFrame(buyHoldSharpes, index=closePriceDF.columns)

#%%save them
     bestFAStrategySummary.to_csv('./train_data/bestFAStrategySummary.csv', sep=';', decimal=',')
     bestFAStrategyParameters.to_csv('./train_data/bestFAStrategyParameters.csv', sep=';', decimal=',')
     bestTAStrategySummary.to_csv('./train_data/bestTAStrategySummary.csv', sep=';', decimal=',')
     bestTAStrategyParameters.to_csv('./train_data/bestTAStrategyParameters.csv', sep=';', decimal=',')
     bHTrainSharpes.to_csv('./train_data/bHTrainSharpes.csv', sep=';', decimal=',')
