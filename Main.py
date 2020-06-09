import pandas as pd
from scipy import stats
import numpy as np
#import math
#import ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import copy
#import itertools
import time
from collections import Counter
from scipy.interpolate import UnivariateSpline
#from statsmodels.tsa import regime_switching
import statsmodels as sm
import seaborn as sns
from datetime import datetime
import time

from days_to_exec import days_left, new_futures_dates, give_first_one, give_first_zero, period_to_hold_options, dates_tuple, get_condition_type1, get_condition_type2, get_condition_type3,get_condition_type4
from rates_interpolation import interpolate_rates
from derivatives import futures_price, Option, CallOpt, PutOpt
from TA_FA import EMA_sign, AD_sign, KAMA_sign, MACD_sign, TRIX_sign
from capital import create_portfolio, sharpe_ratio, calculate_strategies, first_portfolios_with_given_options, portfolios_with_given_options, sharpe_ratios_vector, max_drawdown, mean_return
#import best_portfolios_t_1_to_3 
#import best_portfolios_t_4 
#import best_portfolios_t_5
from condition_types import type1_sharpes, type2_sharpes, type3_sharpes, type4_sharpes, type5_sharpes
from test_portfolios_all_types import capital_for_given_parameters_type1, capital_for_given_parameters_type2, capital_for_given_parameters_type3, capital_for_given_parameters_type4, capital_for_given_parameters_type5
from immitational_modelling import immit_mod_FA_Sharpes, immit_mod_TA_Sharpes


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

countriesListInRussian = ['Канада', 'США', 'Франция', 'Германия', 'Великобр.', 'Австралия', 'Гонконг', 'Япония', 'Бельгия', 'Швейцария', 'Польша', 'Россия', 'Юж. Африка', 'Китай', 'Индия', 'Юж. Корея', 'Греция', 'Бразилия', 'Венгрия', 'Чехия']
countriesListAbbrev = ['CAN', 'USA', 'FRA', 'DEU', 'GBR', 'AUS', 'HKG', 'JPN', 'BEL', 'CHE', 'POL', 'RUS', 'ZAF', 'CHN', 'IND', 'KOR', 'GRC', 'BRA', 'HUN', 'CZE']



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

#%% DATA SET WITH PERIODS OF TRAIN AND TEST PART 
'''
dateTimeList = list(closePriceDF.index)
datesOnlyList = [str(x.date()) for x in dateTimeList]
trainSetStartList = [datesOnlyList[int(ind)] for ind in list(firstObservIndices.iloc[:,0])]
trainSetFinishList = [datesOnlyList[int(ind)-1] for ind in trainSetNumOfObserv]
trainSetPeriod = [trainSetStartList[ind] + ' - ' + trainSetFinishList[ind] for ind in list(range(len(trainSetStartList)))]

testSetStartList = [datesOnlyList[-int(ind)] for ind in testSetNumOfObserv]
testSetFinishList = [datesOnlyList[-1] for ind in testSetNumOfObserv]
testSetPeriod = [testSetStartList[ind] + ' - ' + testSetFinishList[ind] for ind in list(range(len(testSetStartList)))]

periodsDF = pd.DataFrame(zip(trainSetPeriod, testSetPeriod), columns=['Train','Test'], index=closePriceDF.columns)
periodsDF.to_csv('./data/train_test_periods.csv', sep=';')
'''

#%% TRAINING FOR EACH COUNTRY
'''
buyHoldSharpes = [float('nan')]*closePriceDF.shape[1]#all the Sharpe ratios will be stored here
allCountriesSummary = [float('nan')]*closePriceDF.shape[1]#best Sharpes, best strategy numbers, best FA types and values for all the countries will be stored here
allCountriesParameters = [float('nan')]*closePriceDF.shape[1]#best parameters for the FA indicators will be stored here
allCountriesTASummary = [float('nan')]*closePriceDF.shape[1]#best option strategy number and best sharpe ratios for each country
allCountriesTAParameters = [float('nan')]*closePriceDF.shape[1]#best TA parameters with system based only on TA indicators shown
#start_time = time.time()

for countrynum in range(15,closePriceDF.shape[1]):
     #countrynum = 0
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
     daysToRenewInit = new_futures_dates(daysOfExpirationTr, futuresPrice) #list with days when to reconsider position (at this step includes only first day when futures data is available and day after option and futures expiration)
     vixList = list(vix.iloc[:trainLength,countrynum])# vix index for a country in a loop (format: percent per year)

     pBook = list(indPB.iloc[:trainLength,countrynum])
     pEarn = list(indPE.iloc[:trainLength,countrynum])
     pSale = list(indPS.iloc[:trainLength,countrynum])

     summaryMatrix = np.empty((3, 12))#12 strategies with 3 values in output
     parametersMatrix = np.empty((6, 12))#12 strategies with 3 values in output

     summaryMatrix[:,0], parametersMatrix[:,0]  = type1_sharpes(countrynum, pBook, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     summaryMatrix[:,1], parametersMatrix[:,1]  = type1_sharpes(countrynum, pEarn, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     summaryMatrix[:,2], parametersMatrix[:,2]  = type1_sharpes(countrynum, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)

     summaryMatrix[:,3], parametersMatrix[:,3]  = type2_sharpes(countrynum, pBook, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     summaryMatrix[:,4], parametersMatrix[:,4]  = type2_sharpes(countrynum, pEarn, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     summaryMatrix[:,5], parametersMatrix[:,5] = type2_sharpes(countrynum, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)

     summaryMatrix[:,6], parametersMatrix[:,6] = type3_sharpes(countrynum, pBook, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     summaryMatrix[:,7], parametersMatrix[:,7] = type3_sharpes(countrynum, pEarn, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     summaryMatrix[:,8], parametersMatrix[:,8] = type3_sharpes(countrynum, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     
     summaryMatrix[:,9], parametersMatrix[:,9] = type4_sharpes(countrynum, pBook, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     summaryMatrix[:,10], parametersMatrix[:,10] = type4_sharpes(countrynum, pEarn, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     summaryMatrix[:,11], parametersMatrix[:,11] = type4_sharpes(countrynum, pSale, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)
     
     summaryT5, paramsT5  = type5_sharpes(countrynum, spotPrice, clPrSeries, hgPrSeries, lwPrSeries, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, daysOfExpirationTr, daysToRenewInit)

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
     allCountriesTASummary[countrynum] = summaryT5
     allCountriesTAParameters[countrynum] = paramsT5
     
     end = time.time()
     print(f'\nOverall time to calculate: {end - start:.2f}s\n')
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
'''
#%%LOAD files from the training part
bestFAStrategySummary = pd.read_csv('./train_data/bestFAStrategySummary.csv', sep=';', decimal=',', index_col='Unnamed: 0')
bestFAStrategyParameters = pd.read_csv('./train_data/bestFAStrategyParameters.csv', sep=';', decimal=',', index_col='Unnamed: 0')
bestTAStrategySummary = pd.read_csv('./train_data/bestTAStrategySummary.csv', sep=';', decimal=',', index_col='Unnamed: 0')
bestTAStrategyParameters = pd.read_csv('./train_data/bestTAStrategyParameters.csv', sep=';', decimal=',', index_col='Unnamed: 0')
bHTrainSharpes = pd.read_csv('./train_data/bHTrainSharpes.csv', sep=';', decimal=',', index_col='Unnamed: 0')

#%%a bit of analyis
bestFATrainSharpes = list(bestFAStrategySummary.iloc[:,0])
bestTATrainSharpes = list(bestTAStrategySummary.iloc[:,0])
trainBHSharpesList = list(bHTrainSharpes.iloc[:,0])

'''
trainSharpesDF = pd.DataFrame(zip(bestFATrainSharpes, bestTATrainSharpes, trainBHSharpesList), index = countriesListAbbrev, columns=['FA', 'TA', 'BH'])
trainSharpesDeveloped = copy.copy(trainSharpesDF.iloc[:10,:])
trainSharpesEmerging = copy.copy(trainSharpesDF.iloc[10:,:])
trainSharpesDevelopedSorted = trainSharpesDeveloped.sort_values(by=['FA'], ascending=False)
trainSharpesEmergingSorted = trainSharpesEmerging.sort_values(by=['FA'], ascending=False)

ind = np.arange(len(list(bestFAStrategySummary.iloc[:,0])))  #initial x axis (bars are added according to this range)
width = 0.25  # the width of the bars
fig, ax = plt.subplots(figsize=(6.488, 2.81))
fASharpes = ax.bar(ind - width, (list(trainSharpesDevelopedSorted.iloc[:,0])+list(trainSharpesEmergingSorted.iloc[:,0])), width, label='ФА и ТА', color='#6666ff')
tASharpes = ax.bar(ind, (list(trainSharpesDevelopedSorted.iloc[:,1])+list(trainSharpesEmergingSorted.iloc[:,1])), width, label='только ТА', color='#00cc66')
bHBar = ax.bar(ind + width, (list(trainSharpesDevelopedSorted.iloc[:,2])+list(trainSharpesEmergingSorted.iloc[:,2])), width, label='B&H', color='#ff6699')
plt.xticks(ind, (list(trainSharpesDevelopedSorted.index)+list(trainSharpesEmergingSorted.index)), rotation=80)
plt.legend(fontsize=9, loc=0)
plt.ylabel('к. Шарпа')
plt.xlabel('')
plt.show()
fig.savefig('./plots/train_sharpes.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)
#later should be ranged by blue line, and consist of two parts (developed and emerging markets)
'''
#%%analysis tables
#what are the most popular FA strategies
fAAndTypes = list(bestFAStrategySummary.iloc[:,-1])
uniqueFAAndTypes = list(Counter(fAAndTypes).keys())
uniqueFAAndTypesNumber = list(Counter(fAAndTypes).values())
#most popular FA indicators
fAList = []
for i in range(0, len(fAAndTypes)):
     fAList.append(fAAndTypes[i][:2])
uniqueFA = list(Counter(fAList).keys())
uniqueFANumber = list(Counter(fAList).values())
#most popular type fro FA
typesList = []
for i in range(0, len(fAAndTypes)):
     typesList.append(fAAndTypes[i][3:])
uniqueTypes = list(Counter(typesList).keys())
uniqueTypesNumber = list(Counter(typesList).values())
#what is better: FA or TA only system
bestCountrySharpes = [x if x>y else y for x, y in zip(bestFATrainSharpes, bestTATrainSharpes)]
bestCountryStrategyType = ['FA' if x>y else 'TA' for x, y in zip(bestFATrainSharpes, bestTATrainSharpes)]
uniqueFAorTA = list(Counter(bestCountryStrategyType).keys())
uniqueFAorTANumber = list(Counter(bestCountryStrategyType).values())
#average percentage change in Sharpe coefficients
strategies = ['putATM', 'put95', 'strangle5', 'strangle10', 'bearSpread10085', 'bearSpread10080', 'bearSpread9585']
bestOptionStrategyForCountry = [bestFAStrategySummary.iloc[ind,1] if x>y else bestTAStrategySummary.iloc[ind,1] for ind, x, y in zip(list(range(len(bestFAStrategySummary))), bestFATrainSharpes, bestTATrainSharpes)]
bestOptionStrategyNameForCountry = [strategies[int(i)] for i in bestOptionStrategyForCountry]
uniqueOptionStrategy = list(Counter(bestOptionStrategyNameForCountry).keys())
uniqueOptionStrategyNumber = list(Counter(bestOptionStrategyNameForCountry).values())
trainSharpesRateOfChange = [(x-y)/y for x,y in zip(bestCountrySharpes, trainBHSharpesList)]
positiveTrainSharpes = [ind for ind, x  in enumerate(trainBHSharpesList) if x>0]
meanTrainDevelopedImprovement = np.mean([trainSharpesRateOfChange[i] for i in positiveTrainSharpes if i<10])
meanTrainEmergingImprovement = np.mean([trainSharpesRateOfChange[i] for i in positiveTrainSharpes if i>9]) 

#%%test part
testBHSharpesList = [float('nan')]*closePriceDF.shape[1]
testBHDynamics = np.empty((max(testSetNumOfObserv),closePriceDF.shape[1]))
testBHDynamics[:] = np.nan
bestFAPortfoliosDynamics = np.empty((max(testSetNumOfObserv),closePriceDF.shape[1]))
bestFAPortfoliosDynamics[:] = np.nan
testFASharpes = [float('nan')]*closePriceDF.shape[1]
bestEverFATestSharpes = [float('nan')]*closePriceDF.shape[1]
bestTAPortfoliosDynamics = np.empty((max(testSetNumOfObserv),closePriceDF.shape[1]))
bestTAPortfoliosDynamics[:] = np.nan
testTASharpes = [float('nan')]*closePriceDF.shape[1]
bestEverTATestSharpes = [float('nan')]*closePriceDF.shape[1]

for countrynum in range(0,closePriceDF.shape[1]):
     #countrynum = 0
     numOfObserv = testSetNumOfObserv[countrynum]
     testClosePriceSeries = closePriceDF.iloc[:,countrynum]
     testClosePriceList = list(testClosePriceSeries)
     testHighPriceSeries = highPriceDF.iloc[:,countrynum]
     testLowPriceSeries = lowPriceDF.iloc[:,countrynum]
     testInterpRates = list(interpolatedRates.iloc[:,countrynum])
     testDaysTillExp = daysLeft[0]
     testVIXList = list(vix.iloc[:,countrynum])
     testDaysOfExp = daysLeft[1]
     bestFALevel = float(bestFAStrategySummary.iloc[countrynum,2])
     bestFAParams = list(bestFAStrategyParameters.iloc[countrynum,:])
     bestFAOptionStrategy = int(float(bestFAStrategySummary.iloc[countrynum,1]))
     
     bestTAParams = list(bestTAStrategyParameters.iloc[countrynum,:])
     bestTAOptionStrategy = int(float(bestTAStrategySummary.iloc[countrynum,1]))
     
     print('Test part for {}'.format(closePriceDF.columns[countrynum]))
     testBHSharpesList[countrynum] = sharpe_ratio(list(closePriceDF.iloc[:,countrynum])[-numOfObserv:])
     testBHDynamics[-numOfObserv:,countrynum] = testClosePriceList[-numOfObserv:]

     if bestFAStrategySummary.iloc[countrynum,3]=='PBT1' or bestFAStrategySummary.iloc[countrynum,3]=='PBT2' or bestFAStrategySummary.iloc[countrynum,3]=='PBT3':
          bestFA = list(indPB.iloc[:,countrynum])
     elif bestFAStrategySummary.iloc[countrynum,3]=='PET1' or bestFAStrategySummary.iloc[countrynum,3]=='PET2' or bestFAStrategySummary.iloc[countrynum,3]=='PET3':
          bestFA = list(indPE.iloc[:,countrynum])
     else:
          bestFA = list(indPS.iloc[:,countrynum])

     if bestFAStrategySummary.iloc[countrynum,3]=='PBT1' or bestFAStrategySummary.iloc[countrynum,3]=='PET1' or bestFAStrategySummary.iloc[countrynum,3]=='PST1':
          bestFAPortfoliosDynamics[-numOfObserv:,countrynum], testFASharpes[countrynum], bestEverFATestSharpes[countrynum] = capital_for_given_parameters_type1(countrynum, bestFA, numOfObserv, bestFALevel, bestFAParams, bestFAOptionStrategy, testClosePriceList, testClosePriceSeries, testHighPriceSeries, testLowPriceSeries, testInterpRates, testDaysTillExp, testVIXList, testDaysOfExp)
     elif bestFAStrategySummary.iloc[countrynum,3]=='PBT2' or bestFAStrategySummary.iloc[countrynum,3]=='PET2' or bestFAStrategySummary.iloc[countrynum,3]=='PST2':
          bestFAPortfoliosDynamics[-numOfObserv:,countrynum], testFASharpes[countrynum], bestEverFATestSharpes[countrynum] = capital_for_given_parameters_type2(countrynum, bestFA, numOfObserv, bestFALevel, bestFAParams, bestFAOptionStrategy, testClosePriceList, testClosePriceSeries, testHighPriceSeries, testLowPriceSeries, testInterpRates, testDaysTillExp, testVIXList, testDaysOfExp)
     elif bestFAStrategySummary.iloc[countrynum,3]=='PBT3' or bestFAStrategySummary.iloc[countrynum,3]=='PET3' or bestFAStrategySummary.iloc[countrynum,3]=='PST3':
          bestFAPortfoliosDynamics[-numOfObserv:,countrynum], testFASharpes[countrynum], bestEverFATestSharpes[countrynum] = capital_for_given_parameters_type3(countrynum, bestFA, numOfObserv, bestFALevel, bestFAParams, bestFAOptionStrategy, testClosePriceList, testClosePriceSeries, testHighPriceSeries, testLowPriceSeries, testInterpRates, testDaysTillExp, testVIXList, testDaysOfExp)
     else:
          bestFAPortfoliosDynamics[-numOfObserv:,countrynum], testFASharpes[countrynum], bestEverFATestSharpes[countrynum] = capital_for_given_parameters_type4(countrynum, bestFA, numOfObserv, bestFALevel, bestFAParams, bestFAOptionStrategy, testClosePriceList, testClosePriceSeries, testHighPriceSeries, testLowPriceSeries, testInterpRates, testDaysTillExp, testVIXList, testDaysOfExp)

#TA signals calculation
     bestTAPortfoliosDynamics[-numOfObserv:,countrynum], testTASharpes[countrynum], bestEverTATestSharpes[countrynum] = capital_for_given_parameters_type5(countrynum, numOfObserv, bestTAParams, bestTAOptionStrategy, testClosePriceList, testClosePriceSeries, testHighPriceSeries, testLowPriceSeries, testInterpRates, testDaysTillExp, testVIXList, testDaysOfExp)

#%%
testBestCountrySharpes = [x if x>y else y for x, y in zip(testFASharpes, testTASharpes)]
testBestCountryStrategyType = ['FA' if x>y else 'TA' for x, y in zip(testFASharpes, testTASharpes)]
testUniqueFAorTA = list(Counter(testBestCountryStrategyType).keys())
testUniqueFAorTANumber = list(Counter(testBestCountryStrategyType).values())#fa helped not to spend too much capital on hedging when it is not necessary
#avarage percentage change in Sharpe coefficients
testSharpesRateOfChange = [(x-y)/y for x,y in zip(testBestCountrySharpes, testBHSharpesList)]
positiveTestSharpes = [ind for ind, x  in enumerate(testBHSharpesList) if x>0]
meanTestDevelopedImprovement = np.mean([testSharpesRateOfChange[i] for i in positiveTestSharpes if i<10])
meanTestEmergingImprovement = np.mean([testSharpesRateOfChange[i] for i in positiveTestSharpes if i>9])
testSharpesChangePP = [(x-y) for x,y in zip(testBestCountrySharpes, testBHSharpesList)]

#%%data analysis
#two big plots with dynamics of bh and best portfolio
#DEVELOPED MARKETS
'''
fig = plt.figure(figsize=(6.488, 3.7))
fig.subplots_adjust(hspace=0.2, wspace=0.1)
for i in range(0, 10):
     ax = fig.add_subplot(2, 5, i+1)
     if testSharpesChangePP[i]<0:
          ax.set(facecolor = '#ff9999')
     else:
          ax.set(facecolor = 'white')
     ax.plot(testBHDynamics[:,i], lw=0.8, label='B&H', color = '#6666ff')
     if testBestCountryStrategyType[i]=='TA':
          ax.plot(bestTAPortfoliosDynamics[:,i], lw=0.8, label='Портфель', color = '#66cc66')
     else:
          ax.plot(bestFAPortfoliosDynamics[:,i], lw=0.8, label='Портфель', color = '#66cc66')
     ax.set_title(str(countriesListInRussian[i]),fontsize=8)
     ax.set_yticklabels([])
     ax.set_xticklabels([])
     ax.set_yticks([])
     ax.set_xticks([])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=8, fancybox=False, shadow=False,fontsize=8)
fig.savefig('./plots/test_dynamics_developed.jpg', dpi=500, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None) #if bh is better, so the market can be thought as efficient

#EMERGING MARKETS
fig = plt.figure(figsize=(6.488, 3.7))
fig.subplots_adjust(hspace=0.2, wspace=0.1)
for i in range(10, 20):
     ax = fig.add_subplot(2, 5, i-9)
     if testSharpesChangePP[i]<0:
          ax.set(facecolor = '#ff9999')
     else:
          ax.set(facecolor = 'white')
     ax.plot(testBHDynamics[:,i], lw=0.8, label='B&H', color = '#6666ff')
     if testBestCountryStrategyType[i]=='TA':
          ax.plot(bestTAPortfoliosDynamics[:,i], lw=0.8, label='Портфель', color = '#66cc66')
     else:
          ax.plot(bestFAPortfoliosDynamics[:,i], lw=0.8, label='Портфель', color = '#66cc66')
     ax.set_title(str(countriesListInRussian[i]),fontsize=8)
     ax.set_yticklabels([])
     ax.set_xticklabels([])
     ax.set_yticks([])
     ax.set_xticks([])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=8, fancybox=False, shadow=False,fontsize=8)
fig.savefig('./plots/test_dynamics_emerging.jpg', dpi=500, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)
'''
#%% test part analysis
'''
testSharpesDF = pd.DataFrame(zip(testFASharpes, testTASharpes, testBHSharpesList), index = countriesListAbbrev, columns=['FA', 'TA', 'BH'])
testSharpesDeveloped = copy.copy(testSharpesDF.iloc[:10,:])
testSharpesEmerging = copy.copy(testSharpesDF.iloc[10:,:])
testSharpesDevelopedSorted = testSharpesDeveloped.sort_values(by=['FA'], ascending=False)
testSharpesEmergingSorted = testSharpesEmerging.sort_values(by=['FA'], ascending=False)

ind = np.arange(len(list(bHTrainSharpes.iloc[:,0])))  #initial x axis (bars are added according to this range)
width = 0.25  # the width of the bars
fig, ax = plt.subplots(figsize=(6.488, 2.81))
fASharpes = ax.bar(ind - width, (list(testSharpesDevelopedSorted.iloc[:,0])+list(testSharpesEmergingSorted.iloc[:,0])), width, label='ФА и ТА', color='#6666ff')
tASharpes = ax.bar(ind, (list(testSharpesDevelopedSorted.iloc[:,1])+list(testSharpesEmergingSorted.iloc[:,1])), width, label='только ТА', color='#00cc66')
bHBar = ax.bar(ind + width, (list(testSharpesDevelopedSorted.iloc[:,2])+list(testSharpesEmergingSorted.iloc[:,2])), width, label='B&H', color='#ff6699')
plt.xticks(ind, (list(testSharpesDevelopedSorted.index) + list(testSharpesEmergingSorted.index)), rotation=80)
plt.legend(fontsize=9, loc=0)
plt.ylabel('к. Шарпа')
plt.xlabel('')
fig.savefig('./plots/test_sharpes.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)
'''
#%% continue analysis
#change of Sharpe coeffs between train and test samples
bHTrainTestSharpesDelta = [x-y for x,y in zip(testBHSharpesList, trainBHSharpesList)]
bestTrainTestSharpesDelta = [x-y for x,y in zip(testBestCountrySharpes, bestCountrySharpes)]
#countries where optimal system is better than BH 
countriesWithWorkingStrategy = [ind for ind in list(range(len(testBestCountrySharpes))) if testBestCountrySharpes[ind]>testBHSharpesList[ind]]
meanBHSharpesDelta = np.mean([bHTrainTestSharpesDelta[i] for i in countriesWithWorkingStrategy])
meanBestSharpesDelta = np.mean([bestTrainTestSharpesDelta[i] for i in countriesWithWorkingStrategy]) #lost more due to overfitting - train results are better than test
#best strategies for developed/emerging countries
bestStrategyForCountries = [testBestCountryStrategyType[i] if testBestCountrySharpes[i] > testBHSharpesList[i] else 'BH' for i in list(range(len(testBestCountrySharpes)))]
bestStrategyForDeveloped = bestStrategyForCountries[:10]
bestStrategyForEmerging = bestStrategyForCountries[10:]
testUniqueStrategiesForDeveloped = list(Counter(bestStrategyForDeveloped).keys())
testUniqueStrategiesForDevelopedNumber = list(Counter(bestStrategyForDeveloped).values())
#testUniqueStrategiesForDeveloped.append('TA')
#testUniqueStrategiesForDevelopedNumber.append(0)
testUniqueStrategiesForEmerging = list(Counter(bestStrategyForEmerging).keys())
testUniqueStrategiesForEmergingNumber = list(Counter(bestStrategyForEmerging).values())
testUniqueStrategiesForEmerging.append('BH')
testUniqueStrategiesForEmergingNumber.append(0)
testUniqueStrategiesForEmerging[1], testUniqueStrategiesForEmerging[2] = testUniqueStrategiesForEmerging[2], testUniqueStrategiesForEmerging[1]
testUniqueStrategiesForEmergingNumber[1], testUniqueStrategiesForEmergingNumber[2] = testUniqueStrategiesForEmergingNumber[2], testUniqueStrategiesForEmergingNumber[1]

testBestStrategySummary = pd.DataFrame([testUniqueStrategiesForDevelopedNumber, testUniqueStrategiesForEmergingNumber], index=['Developed','Emerging'], columns=testUniqueStrategiesForEmerging) #developed countries can be assumed more efficient. Everywhere FA makes impact

#%%
#look at maximum drawdowns for HB ad best strategies (plot with max drawdown for best strategy according to max drawdown for BH )
bHDradownsList = []
bestPortfDradownsList = []
for i in range(len(testBHSharpesList)):
     bHDradownsList.append(max_drawdown(testBHDynamics[:,i]))
     if testBestCountryStrategyType[i]=='TA':
          bestPortfDradownsList.append(max_drawdown(bestTAPortfoliosDynamics[:,i]))
     else:
          bestPortfDradownsList.append(max_drawdown(bestFAPortfoliosDynamics[:,i]))

drawdownsDF = pd.DataFrame(zip(bHDradownsList, bestPortfDradownsList), columns=['BH', 'Best_strategy'])
drawdownsDFSorted = drawdownsDF.sort_values(by=['BH'], ascending=True)
sortedBHDrawdowns = list(drawdownsDFSorted.iloc[:,0])
sortedPortfolioDrawdowns = list(drawdownsDFSorted.iloc[:,1])
spl = UnivariateSpline(sortedBHDrawdowns, sortedPortfolioDrawdowns)
sortedBHDrawSmooth = np.linspace(np.min(sortedBHDrawdowns), np.max(sortedBHDrawdowns), 200)
sortedPortfDrawSmooth = spl(sortedBHDrawSmooth)
'''
fig, ax = plt.subplots(figsize=(6.488, 2.81))
ax.plot([x*100 for x in sortedBHDrawSmooth], [x*100 for x in sortedPortfDrawSmooth], label='Просадки по портфелям', color='#6666ff', lw=1.5)
ax.plot([x*100 for x in sortedBHDrawSmooth], [x*100 for x in sortedBHDrawSmooth], label='Биссектриса', color='#999999', lw=1.5)
ax.set_ylim(12,40)
ax.set_xlim(12,65)
ax.set_ylabel('Просадки по портфелям, %')
ax.set_xlabel('Прсадки по B&H, %')
ax.legend(loc=2, fontsize=9)
fig.savefig('./plots/drawdowns.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)
#'''
#average drawdown decline (if a decline, so the value is positive)
drawdownsRateOfChange = [-100*(x-y)/y for x,y in zip(bestPortfDradownsList, bHDradownsList)]
meanDrawdownRateOfChange = np.mean(drawdownsRateOfChange)# on average maximum drawdowns are almost quarter less in best portfolios, than in BH

#%%
#is portfolio efficiency correlated with maximum drawdows of BH?
drawdownsVSImprovementDF = pd.DataFrame(zip(bHDradownsList, testSharpesChangePP), columns=['Drawdown', 'Improvement'])
drawdownsVSImprovementDFSorted = drawdownsVSImprovementDF.sort_values(by=['Drawdown'], ascending=True)
sortedBHDrawdowns = list(drawdownsVSImprovementDFSorted.iloc[:,0])
sortedImprovement = list(drawdownsVSImprovementDFSorted.iloc[:,1])
spl = UnivariateSpline(sortedBHDrawdowns, sortedImprovement)
spl.set_smoothing_factor(0.14)
sortedBHDrawSmooth = np.linspace(np.min(sortedBHDrawdowns), np.max(sortedBHDrawdowns), 200)
sortedImprovementSmooth = spl(sortedBHDrawSmooth)
'''
fig, ax = plt.subplots(figsize=(6.488, 2.81))
ax.plot([x*100 for x in sortedBHDrawSmooth], [x*100 for x in sortedImprovementSmooth], label='Просадки по портфелям', color='#6666ff', lw=1.5)
#ax.plot([x*100 for x in sortedBHDrawSmooth], [x*100 for x in sortedBHDrawSmooth], label='Биссектриса', color='#999999', lw=1.5)
#ax.set_ylim(12,40)
#ax.set_xlim(12,65)
ax.set_ylabel('Прирост к. Шарпа, п. п.')
ax.set_xlabel('Просадки по B&H, %')
#ax.legend(loc=2, fontsize=9)
fig.savefig('./plots/improvement_vs_drawdown.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None) #there is level of drawdown, where sharpe ratio improvement  rises with incresing drawdown from BH
#'''
#%%
#mean returns analysis

bHMeanReturnsList = []
bestPortfMeanReturnsList = []
for i in range(len(testBHSharpesList)):
     bHMeanReturnsList.append(mean_return(testBHDynamics[:,i]))
     if testBestCountryStrategyType[i]=='TA':
          bestPortfMeanReturnsList.append(mean_return(bestTAPortfoliosDynamics[:,i]))
     else:
          bestPortfMeanReturnsList.append(mean_return(bestFAPortfoliosDynamics[:,i]))

meanReturnsDF = pd.DataFrame(zip(bHMeanReturnsList, bestPortfMeanReturnsList), columns=['BH', 'Best_strategy'])
meanReturnsDFSorted = meanReturnsDF.sort_values(by=['BH'], ascending=True)
sortedBHMeanReturns = list(meanReturnsDFSorted.iloc[:,0])
sortedPortfolioMeanReturns = list(meanReturnsDFSorted.iloc[:,1])
spl = UnivariateSpline(sortedBHMeanReturns, sortedPortfolioMeanReturns)
spl.set_smoothing_factor(0.09)
sortedBHRetSmooth = np.linspace(np.min(sortedBHMeanReturns), np.max(sortedBHMeanReturns), 200)
sortedPortfRetSmooth = spl(sortedBHMeanReturns)
'''
fig, ax = plt.subplots(figsize=(6.488, 2.81))
ax.plot([x*100 for x in sortedBHMeanReturns], [x*100 for x in sortedPortfRetSmooth], label='Доходность по портфелям', color='#6666ff', lw=1.5)
ax.plot([x*100 for x in sortedBHMeanReturns], [x*100 for x in sortedBHMeanReturns], label='Биссектриса', color='#999999', lw=1.5)
#ax.set_ylim(12,40)
#ax.set_xlim(12,65)
ax.set_ylabel('Доходность портфелей, % в год')
ax.set_xlabel('Доходность B&H, % в год')
ax.legend(loc=2, fontsize=9)
fig.savefig('./plots/mean_returns.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)#strategy saves capital when BH shows negative return and it does not lose money when BH returns are high and positive
'''
#average mean return change p.p.
meanReturnChangePP = [100*(x-y) for x,y in zip(bestPortfMeanReturnsList, bHMeanReturnsList)]
meanRetChangePP = np.mean(meanReturnChangePP)
maxRetChangePP = np.max(meanReturnChangePP)
minRetChangePP = np.min(meanReturnChangePP)
numOfCountriesWithPositiveMeanRetChange = len([ind for ind, x in enumerate(meanReturnChangePP) if meanReturnChangePP[ind]>0])# only 13 countries with positive change of return (but 16 countries with better Sharpe ratio), so system also reduces volatility

#%%
#comparison with riskfree
riskFreeList = []
for i in range(len(testBHSharpesList)):
     rates3MTest = list(rates3M.iloc[-testSetNumOfObserv[i]:,i])
     logRates3MTest = [np.log(1+x/100) for x in rates3MTest]
     mean3MRate = np.exp(np.nanmean(logRates3MTest))-1
     riskFreeList.append(mean3MRate)

#number of countries where portfolio gives better return than risk free
portfolioVSRiskFreeChangePP = [100*(x-y) for x,y in zip(bestPortfMeanReturnsList, riskFreeList)]
numOfCountrieWithPositivePortfolioToRiskFree = len([ind for ind,x in enumerate(portfolioVSRiskFreeChangePP) if portfolioVSRiskFreeChangePP[ind]>0])

#number of countries where BH gives better return than risk free
bHVSRiskFreeChangePP = [100*(x-y) for x,y in zip(bHMeanReturnsList, riskFreeList)]
numOfCountrieWithPositiveBHToRiskFree = len([ind for ind,x in enumerate(bHVSRiskFreeChangePP) if bHVSRiskFreeChangePP[ind]>0])

#%% IMITATIONAL MODELLING
#try on USA - countrynum=1
countrynum=1
numOfObservUSA = testSetNumOfObserv[countrynum]
closePriceSeriesUSA = closePriceDF.iloc[:,countrynum]
closePriceListUSA = list(closePriceSeriesUSA)
highPriceSeriesUSA = highPriceDF.iloc[:,countrynum]
lowPriceSeriesUSA = lowPriceDF.iloc[:,countrynum]
interpRatesUSA = list(interpolatedRates.iloc[:,countrynum])
daysTillExpUSA = daysLeft[0]
vIXListUSA = list(vix.iloc[:,countrynum])
daysOfExpUSA = daysLeft[1]
fALevelUSA = float(bestFAStrategySummary.iloc[countrynum,2])
fAParamsUSA = list(bestFAStrategyParameters.iloc[countrynum,:])
fAOptionStrategyUSA = int(float(bestFAStrategySummary.iloc[countrynum,1]))
faTypeOfFAUSA = bestFAStrategySummary.iloc[countrynum,3] #check for a country
bestFAUSA = list(indPE.iloc[:,countrynum])#check for a country
tAParamsUSA = list(bestTAStrategyParameters.iloc[countrynum,:])
tAOptionStrategyUSA = int(float(bestTAStrategySummary.iloc[countrynum,1]))#check for a country

earningsLevelUSA = [x/y for x,y in zip(closePriceListUSA, bestFAUSA)]         
#plt.plot(earningsLevelUSA[-numOfObservUSA:])
logPricesUSA = [np.log(x) for x in closePriceListUSA]
logPriceUSATillTestObserv = logPricesUSA[:trainSetNumOfObserv[countrynum]]
#x = logPricesUSA[-testSetNumOfObserv[1]]
#y = logPricesUSA[-testSetNumOfObserv[1]:]
returnsUSA = list(closePriceSeriesUSA.iloc[-numOfObservUSA:].pct_change())
logReturnsUSA = [np.log(1+x) for x in returnsUSA]
returnsVectorLength = len(list(pd.Series(logReturnsUSA).dropna()))
meanReturnUSA = np.nanmean(logReturnsUSA)
sDReturnUSA = np.std(pd.Series(logReturnsUSA).dropna())

'''
immitNormFASharpesList = []
immitNormBHSharpesList = []
immitNormTASharpesList = []
np.random.seed(10)
start = time.time()
for i in range(1000):
     if i%100==0 and i>0:
          print('{} iterations completed'.format(i))
     normalGeneratedUSA = np.random.normal(loc=meanReturnUSA, scale=sDReturnUSA, size=returnsVectorLength)
     logPriceNormalGeneratedTestSet = []
     logPriceNormalGeneratedTestSet.append(logPricesUSA[-testSetNumOfObserv[countrynum]])
     for i in range(len(normalGeneratedUSA)):
          newLogPrice = logPriceNormalGeneratedTestSet[-1]+normalGeneratedUSA[i]
          logPriceNormalGeneratedTestSet.append(newLogPrice)

     logPriceNormalGeneratedOverall = logPriceUSATillTestObserv+logPriceNormalGeneratedTestSet
     closePriceNormalGeneratedOverallList = [np.exp(x) for x in logPriceNormalGeneratedOverall]
     closePriceNormalGeneratedOverallSeries = pd.Series(closePriceNormalGeneratedOverallList)
     highPriceNormalGeneratedOverallSeries = pd.Series([x*(y/z) for x, y, z in zip(closePriceNormalGeneratedOverallList, list(highPriceSeriesUSA), closePriceListUSA)])
     lowPriceNormalGeneratedOverallSeries = pd.Series([x*(y/z) for x, y, z in zip(closePriceNormalGeneratedOverallList, list(lowPriceSeriesUSA), closePriceListUSA)])

     generatedNormFAindicator = [x/y for x,y in zip(closePriceNormalGeneratedOverallList, earningsLevelUSA)]

     generatedNormFASharpe = immit_mod_FA_Sharpes(countrynum, generatedNormFAindicator, numOfObservUSA, fALevelUSA, fAParamsUSA, closePriceNormalGeneratedOverallList, closePriceNormalGeneratedOverallSeries, highPriceNormalGeneratedOverallSeries, lowPriceNormalGeneratedOverallSeries, interpRatesUSA, daysTillExpUSA, vIXListUSA, daysOfExpUSA)
     generatedNormTASharpe = immit_mod_TA_Sharpes(countrynum, numOfObservUSA, tAParamsUSA, closePriceNormalGeneratedOverallList, closePriceNormalGeneratedOverallSeries, highPriceNormalGeneratedOverallSeries, lowPriceNormalGeneratedOverallSeries, interpRatesUSA, daysTillExpUSA, vIXListUSA, daysOfExpUSA)

     immitNormFASharpesList.append(generatedNormFASharpe)
     immitNormTASharpesList.append(generatedNormTASharpe)
     immitNormBHSharpesList.append(sharpe_ratio(closePriceNormalGeneratedOverallList[-numOfObservUSA:])) 
end = time.time()
print(f'\nOverall time to calculate: {(end - start)/60:.2f}m\n')

simulationsNormDF = pd.DataFrame(zip(immitNormFASharpesList, immitNormTASharpesList, immitNormBHSharpesList), columns=['FA','TA','BH'])
simulationsNormDF.to_csv('./simulations/simulationsNormDF.csv', sep=';', decimal=',')
#'''
'''
#%%
#save USA log returns for simulations
logRetUSADF = pd.DataFrame(logReturnsUSA)
logRetUSADF.to_csv('./simulations/logRetUSADF.csv', sep=';', decimal=',')
#send this file to RStudio to get simulations (poor, but could not implement it in python)

#markov switching model simulations (using file from RStudio)
simulMSLogRetUSA = pd.read_csv('./simulations/simulationsMSLogRetDF.csv', sep=',', decimal='.', index_col='Unnamed: 0')
'''
#%% run simulations
'''
immitMSFASharpesList = []
immitMSBHSharpesList = []
immitMSTASharpesList = []
start = time.time()
for i in range(1000):
     if i%100==0 and i>0:
          print('{} iterations completed'.format(i))
     logPriceMSGeneratedTestSet = []
     logPriceMSGeneratedTestSet.append(logPricesUSA[-testSetNumOfObserv[1]])
     for j in range(len(simulMSLogRetUSA)):
          newLogPrice = logPriceMSGeneratedTestSet[-1]+simulMSLogRetUSA.iloc[j,i]
          logPriceMSGeneratedTestSet.append(newLogPrice)

     logPriceMSGeneratedOverall = logPriceUSATillTestObserv+logPriceMSGeneratedTestSet
     closePriceMSGeneratedOverallList = [np.exp(x) for x in logPriceMSGeneratedOverall]
     closePriceMSGeneratedOverallSeries = pd.Series(closePriceMSGeneratedOverallList)
     highPriceMSGeneratedOverallSeries = pd.Series([x*(y/z) for x, y, z in zip(closePriceMSGeneratedOverallList, list(highPriceSeriesUSA), closePriceListUSA)])
     lowPriceMSGeneratedOverallSeries = pd.Series([x*(y/z) for x, y, z in zip(closePriceMSGeneratedOverallList, list(lowPriceSeriesUSA), closePriceListUSA)])

     generatedMSFAindicator = [x/y for x,y in zip(closePriceMSGeneratedOverallList, earningsLevelUSA)]

     generatedMSFASharpe = immit_mod_FA_Sharpes(countrynum, generatedMSFAindicator, numOfObservUSA, fALevelUSA, fAParamsUSA, closePriceMSGeneratedOverallList, closePriceMSGeneratedOverallSeries, highPriceMSGeneratedOverallSeries, lowPriceMSGeneratedOverallSeries, interpRatesUSA, daysTillExpUSA, vIXListUSA, daysOfExpUSA)
     generatedMSTASharpe = immit_mod_TA_Sharpes(countrynum, numOfObservUSA, tAParamsUSA, closePriceMSGeneratedOverallList, closePriceMSGeneratedOverallSeries, highPriceMSGeneratedOverallSeries, lowPriceMSGeneratedOverallSeries, interpRatesUSA, daysTillExpUSA, vIXListUSA, daysOfExpUSA)

     immitMSFASharpesList.append(generatedMSFASharpe)
     immitMSTASharpesList.append(generatedMSTASharpe)
     immitMSBHSharpesList.append(sharpe_ratio(closePriceMSGeneratedOverallList[-numOfObservUSA:])) 
end = time.time()
print(f'\nOverall time to calculate: {(end - start)/60:.2f}m\n')

simulationsMSDF = pd.DataFrame(zip(immitMSFASharpesList, immitMSTASharpesList, immitMSBHSharpesList), columns=['FA','TA','BH'])
simulationsMSDF.to_csv('./simulations/simulationsMSDF.csv', sep=';', decimal=',')
'''

#%%LOAD simulated data
simulationsNormDF = pd.read_csv('./simulations/simulationsNormDF.csv', sep=';', decimal=',', index_col='Unnamed: 0')
simulationsMSDF = pd.read_csv('./simulations/simulationsMSDF.csv', sep=';', decimal=',', index_col='Unnamed: 0')

immitNormFASharpesList = list(simulationsNormDF.iloc[:,0])
immitNormTASharpesList = list(simulationsNormDF.iloc[:,1])
immitNormBHSharpesList = list(simulationsNormDF.iloc[:,2])

immitMSFASharpesList = list(simulationsMSDF.iloc[:,0])
immitMSTASharpesList = list(simulationsMSDF.iloc[:,1])
immitMSBHSharpesList = list(simulationsMSDF.iloc[:,2])

meanNormSimul = [np.nanmean(immitNormFASharpesList), np.nanmean(immitNormTASharpesList), np.nanmean(immitNormBHSharpesList)]
meanMSSimul = [np.nanmean(immitMSFASharpesList), np.nanmean(immitMSTASharpesList), np.nanmean(immitMSBHSharpesList)]
realUSASharpes = [testFASharpes[countrynum], testTASharpes[countrynum], testBHSharpesList[countrynum]]

#%%plot with distributions by model
'''
fig = plt.figure(figsize=(6.488, 2.81))
fig.subplots_adjust(wspace=0)
sns.set(style="white", rc={"lines.linewidth": 1.5})

ax = fig.add_subplot(1,2,1)
a = sns.distplot(immitNormFASharpesList, color="#6666ff", label="ФА и ТА", hist=False)
b = sns.distplot(immitNormTASharpesList, color="#00cc66", label="только ТА", hist=False)
c = sns.distplot(immitNormBHSharpesList, color="#ff6699", label="B&H", hist=False)
c.set_yticklabels([])
c.set_yticks([]) 
c.grid(b=True, linestyle='--')
#c.legend(bbox_to_anchor=[0.6,0.1])
c.get_legend().remove()
c.set_title('Случайное блуждание')
c.set_xlabel('к. Шарпа')

ax = fig.add_subplot(1,2,2)
#ax.set_xlim(-1.1,1.8)
d = sns.distplot(immitMSFASharpesList, color="#6666ff", label="ФА и ТА", hist=False)
e = sns.distplot(immitMSTASharpesList, color="#00cc66", label="только ТА", hist=False)
f = sns.distplot(immitMSBHSharpesList, color="#ff6699", label="B&H", hist=False)
f.set_yticklabels([])
f.set_yticks([]) 
f.grid(b=True, linestyle='--')
f.set_title('Переключения Маркова')
f.set_xlabel('к. Шарпа')
f.legend(loc=1, fontsize=8)

handles, labels = f.get_legend_handles_labels()
fig.savefig('./plots/distributions_by_processes.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)
'''

#%%
'''
fig = plt.figure(figsize=(6.488, 2.81))
fig.subplots_adjust(wspace=0.05)
sns.set(style="white", rc={"lines.linewidth": 1.5})

ax = fig.add_subplot(1,3,1)
ax.axvline(realUSASharpes[0], color='#999999', linestyle='--', label='Реал. данн.')
a = sns.distplot(immitNormFASharpesList, color="#6666ff", label="Сл. блужд.", hist=False)
d = sns.distplot(immitMSFASharpesList, color="#00cc66", label="Перекл. реж.", hist=False)
d.set_yticklabels([])
d.set_yticks([]) 
d.get_legend().remove()
d.set_title('ФА + ТА')
d.set_xlabel('')

ax = fig.add_subplot(1,3,2)
ax.axvline(realUSASharpes[1], color='#999999', linestyle='--', label='Реал. данн.')                 
b = sns.distplot(immitNormTASharpesList, color="#6666ff", label="Сл. блужд.", hist=False)
e = sns.distplot(immitMSTASharpesList, color="#00cc66", label="Перекл. реж.", hist=False)
e.set_yticklabels([])
e.set_yticks([]) 
e.get_legend().remove()
e.set_title('Только ТА')
e.set_xlabel('к. Шарпа')
 
ax = fig.add_subplot(1,3,3)
ax.axvline(realUSASharpes[2], color='#999999', linestyle='--', label='Реал. данн.') 
c = sns.distplot(immitNormBHSharpesList, color="#6666ff", label="Сл. блужд.", hist=False)
f = sns.distplot(immitMSBHSharpesList, color="#00cc66", label="Перекл. реж.", hist=False)                 
f.set_yticklabels([])
f.set_yticks([]) 
f.get_legend().remove()
f.set_title('B&H')
f.set_xlabel('')  

handles, labels = f.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=[0.36,0.95], fancybox=False, shadow=False,fontsize=8, framealpha=1)
fig.savefig('./plots/distributions_by_strategies.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)
'''
#%%
#check quantiles for BH, FA and TA in simulated data for real results for USA
quantileInSimulatedBH = [len([ind for ind, x in enumerate(immitNormBHSharpesList) if immitNormBHSharpesList[ind]<=realUSASharpes[2]])/10, len([ind for ind, x in enumerate(immitMSBHSharpesList) if immitMSBHSharpesList[ind]<=realUSASharpes[2]])/10]
quantileInSimulatedFA = [len([ind for ind, x in enumerate(immitNormFASharpesList) if immitNormFASharpesList[ind]<=realUSASharpes[0]])/10, len([ind for ind, x in enumerate(immitMSFASharpesList) if immitMSFASharpesList[ind]<=realUSASharpes[0]])/10]
quantileInSimulatedTA = [len([ind for ind, x in enumerate(immitNormTASharpesList) if immitNormTASharpesList[ind]<=realUSASharpes[1]])/10, len([ind for ind, x in enumerate(immitMSTASharpesList) if immitMSTASharpesList[ind]<=realUSASharpes[1]])/10]
quantilesDF = pd.DataFrame([quantileInSimulatedFA, quantileInSimulatedTA, quantileInSimulatedBH], columns=['RW','MS'], index=['FA','TA','BH'])#overall, results are better for MS because share of simulated observations that are less than actual values is closer to 50%. It is not the case in FA because behaviour of FA indicators is not changed

#%% TWO SAMPLE MEAN TESTS
immitNormFASharpesListMean = np.mean(immitNormFASharpesList)
immitNormFASharpesListStd = np.std(immitNormFASharpesList)
immitMSFASharpesListMean = np.mean(immitMSFASharpesList)
immitMSFASharpesListStd = np.std(immitMSFASharpesList)

immitNormTASharpesListMean = np.mean(immitNormTASharpesList)
immitNormTASharpesListStd = np.std(immitNormTASharpesList)
immitMSTASharpesListMean = np.mean(immitMSTASharpesList)
immitMSTASharpesListStd = np.std(immitMSTASharpesList)

immitMSBHSharpesListMean = np.mean(immitMSBHSharpesList)
immitMSBHSharpesListStd = np.std(immitMSBHSharpesList)
immitNormBHSharpesListMean = np.mean(immitNormBHSharpesList)
immitNormBHSharpesListStd = np.std(immitNormBHSharpesList)
size = len(immitMSTASharpesList)

tStatsFA = (immitMSFASharpesListMean - immitNormFASharpesListMean)/np.sqrt(immitMSFASharpesListStd**2/size + immitNormFASharpesListStd**2/size)
tStatsTA = (immitMSTASharpesListMean - immitNormTASharpesListMean)/np.sqrt(immitMSTASharpesListStd**2/size + immitNormTASharpesListStd**2/size)
tStatsBH = (immitMSBHSharpesListMean - immitNormBHSharpesListMean)/np.sqrt(immitMSBHSharpesListStd**2/size + immitNormBHSharpesListStd**2/size)

degreesOfFreedomFA = int((immitMSFASharpesListStd**2/size+immitNormFASharpesListStd**2/size)**2/((immitMSFASharpesListStd**2/size)**2/(size-1) + (immitNormFASharpesListStd**2/size)**2/(size-1)))
degreesOfFreedomTA = int((immitMSTASharpesListStd**2/size+immitNormTASharpesListStd**2/size)**2/((immitMSTASharpesListStd**2/size)**2/(size-1) + (immitNormTASharpesListStd**2/size)**2/(size-1)))
degreesOfFreedomBH = int((immitMSBHSharpesListStd**2/size+immitNormBHSharpesListStd**2/size)**2/((immitMSBHSharpesListStd**2/size)**2/(size-1) + (immitNormBHSharpesListStd**2/size)**2/(size-1)))

criticalValueFA = stats.t.ppf(1-0.005, degreesOfFreedomFA)
criticalValueTA = stats.t.ppf(1-0.005, degreesOfFreedomTA)
criticalValueBH = stats.t.ppf(1-0.005, degreesOfFreedomBH)

nullHypoth = [tStatsFA < criticalValueFA, tStatsTA < criticalValueTA, tStatsBH < criticalValueBH]

#%% what is with the newes data - did it start to hedge???
dataTill2020 = pd.read_csv('./data/data_till_2020.csv', sep=';', parse_dates=['Date'], dayfirst=True, index_col='Date', decimal=',')

#%%signals for SP - best is FA strategy
'''
newSPEMA = EMA_sign(dataTill2020.iloc[:,2], int(bestFAStrategyParameters.iloc[1,0]), int(bestFAStrategyParameters.iloc[1,1]))
newSPAD = AD_sign(dataTill2020.iloc[:,0], dataTill2020.iloc[:,1], 5, 34, int(bestFAStrategyParameters.iloc[1,2]))
newSPKAMA = KAMA_sign(dataTill2020.iloc[:,2], 10, 2, 30, int(bestFAStrategyParameters.iloc[1,3]))
newSPMACD = MACD_sign(dataTill2020.iloc[:,2], 26, 12, int(bestFAStrategyParameters.iloc[1,4]))
newSPTRIX = TRIX_sign(dataTill2020.iloc[:,2], int(bestFAStrategyParameters.iloc[1,5]))
wholeSPHedge = [x*y*z*v*w for x, y, z, v, w in zip(newSPEMA, newSPAD, newSPKAMA, newSPMACD, newSPTRIX)]
wholeSPOutOfHedge = [x+y+z+v+w for x, y, z, v, w in zip(newSPEMA, newSPAD, newSPKAMA, newSPMACD, newSPTRIX)]

sPEMAFirstOne = give_first_one(newSPEMA)
sPADFirstOne = give_first_one(newSPAD)
sPKAMAFirstOne = give_first_one(newSPKAMA)
sPMACDFirstOne = give_first_one(newSPMACD)
sPTRIXFirstOne = give_first_one(newSPTRIX)
wholeSPHedgeFirstOne = give_first_one(wholeSPHedge)

sPEMAFirstOneIndex = [ind for ind,x in enumerate(sPEMAFirstOne) if sPEMAFirstOne[ind]==1][-1]
sPADFirstOneIndex = [ind for ind,x in enumerate(sPADFirstOne) if sPADFirstOne[ind]==1][-1]
sPKAMAFirstOneIndex = [ind for ind,x in enumerate(sPKAMAFirstOne) if sPKAMAFirstOne[ind]==1][-1]
sPMACDFirstOneIndex = [ind for ind,x in enumerate(sPMACDFirstOne) if sPMACDFirstOne[ind]==1][-1]
sPTRIXFirstOneIndex = [ind for ind,x in enumerate(sPTRIXFirstOne) if sPTRIXFirstOne[ind]==1][-1]
wholeSPHedgeFirstOneIndex = [ind for ind,x in enumerate(wholeSPHedgeFirstOne) if wholeSPHedgeFirstOne[ind]==1][-1]

sPEMAFirstZero = give_first_zero(newSPEMA)
sPADFirstZero = give_first_zero(newSPAD)
sPKAMAFirstZero = give_first_zero(newSPKAMA)
sPMACDFirstZero = give_first_zero(newSPMACD)
sPTRIXFirstZero = give_first_zero(newSPTRIX)
wholeSPHedgeFirstZero = give_first_zero(wholeSPOutOfHedge)

#sPEMAFirstZeroIndex = [ind for ind,x in enumerate(sPEMAFirstZero) if sPEMAFirstZero[ind]==1][-1]
sPADFirstZeroIndex = [ind for ind,x in enumerate(sPADFirstZero) if sPADFirstZero[ind]==1][-1]
#sPKAMAFirstZeroIndex = [ind for ind,x in enumerate(sPKAMAFirstZero) if sPKAMAFirstZero[ind]==1][-1]
sPMACDFirstZeroIndex = [ind for ind,x in enumerate(sPMACDFirstZero) if sPMACDFirstZero[ind]==1][-1]
#sPTRIXFirstZeroIndex = [ind for ind,x in enumerate(sPTRIXFirstZero) if sPTRIXFirstZero[ind]==1][-1]
#wholeSPHedgeFirstZeroIndex = [ind for ind,x in enumerate(wholeSPHedgeFirstZero) if wholeSPHedgeFirstZero[ind]==1][-1]

#%%signals for RTSI - best is TA strategy
newRTSIEMA = EMA_sign(dataTill2020.iloc[:,5], int(bestTAStrategyParameters.iloc[11,0]), int(bestTAStrategyParameters.iloc[11,1]))
newRTSIAD = AD_sign(dataTill2020.iloc[:,3], dataTill2020.iloc[:,4], 5, 34, int(bestTAStrategyParameters.iloc[11,2]))
newRTSIKAMA = KAMA_sign(dataTill2020.iloc[:,5], 10, 2, 30, int(bestTAStrategyParameters.iloc[11,3]))
newRTSIMACD = MACD_sign(dataTill2020.iloc[:,5], 26, 12, int(bestTAStrategyParameters.iloc[11,4]))
newRTSITRIX = TRIX_sign(dataTill2020.iloc[:,5], int(bestTAStrategyParameters.iloc[11,5]))
wholeRTSIHedge = [x*y*z*v*w for x, y, z, v, w in zip(newRTSIEMA, newRTSIAD, newRTSIKAMA, newRTSIMACD, newRTSITRIX)]
wholeRTSIOutOfHedge = [x+y+z+v+w for x, y, z, v, w in zip(newRTSIEMA, newRTSIAD, newRTSIKAMA, newRTSIMACD, newRTSITRIX)]

rTSIEMAFirstOne = give_first_one(newRTSIEMA)
rTSIADFirstOne = give_first_one(newRTSIAD)
rTSIKAMAFirstOne = give_first_one(newRTSIKAMA)
rTSIMACDFirstOne = give_first_one(newRTSIMACD)
rTSITRIXFirstOne = give_first_one(newRTSITRIX)
wholeRTSIHedgeFirstOne = give_first_one(wholeRTSIHedge)

rTSIEMAFirstOneIndex = [ind for ind,x in enumerate(rTSIEMAFirstOne) if rTSIEMAFirstOne[ind]==1][-1]
rTSIADFirstOneIndex = [ind for ind,x in enumerate(rTSIADFirstOne) if rTSIADFirstOne[ind]==1][-1]
rTSIKAMAFirstOneIndex = [ind for ind,x in enumerate(rTSIKAMAFirstOne) if rTSIKAMAFirstOne[ind]==1][-1]
rTSIMACDFirstOneIndex = [ind for ind,x in enumerate(rTSIMACDFirstOne) if rTSIMACDFirstOne[ind]==1][-1]
rTSITRIXFirstOneIndex = [ind for ind,x in enumerate(rTSITRIXFirstOne) if rTSITRIXFirstOne[ind]==1][-1]
wholeRTSIHedgeFirstOneIndex = [ind for ind,x in enumerate(wholeRTSIHedgeFirstOne) if wholeRTSIHedgeFirstOne[ind]==1][-1]

rTSIEMAFirstZero = give_first_zero(newRTSIEMA)
rTSIADFirstZero = give_first_zero(newRTSIAD)
rTSIKAMAFirstZero = give_first_zero(newRTSIKAMA)
rTSIMACDFirstZero = give_first_zero(newRTSIMACD)
rTSITRIXFirstZero = give_first_zero(newRTSITRIX)
wholeRTSIHedgeFirstZero = give_first_zero(wholeRTSIOutOfHedge)

#rTSIEMAFirstZeroIndex = [ind for ind,x in enumerate(rTSIEMAFirstZero) if rTSIEMAFirstZero[ind]==1][-1]
rTSIADFirstZeroIndex = [ind for ind,x in enumerate(rTSIADFirstZero) if rTSIADFirstZero[ind]==1][-1]
#rTSIKAMAFirstZeroIndex = [ind for ind,x in enumerate(rTSIKAMAFirstZero) if rTSIKAMAFirstZero[ind]==1][-1]
#rTSIMACDFirstZeroIndex = [ind for ind,x in enumerate(rTSIMACDFirstZero) if rTSIMACDFirstZero[ind]==1][-1]
#rTSITRIXFirstZeroIndex = [ind for ind,x in enumerate(rTSITRIXFirstZero) if rTSITRIXFirstZero[ind]==1][-1]
#wholeRTSIHedgeFirstZeroIndex = [ind for ind,x in enumerate(wholeRTSIHedgeFirstZero) if wholeRTSIHedgeFirstZero[ind]==1][-1]

#%%
months = mdates.MonthLocator()  

fig = plt.figure(figsize=(6.488, 2.81))
fig.subplots_adjust(wspace=0.3)

ax = fig.add_subplot(1,2,1)
ax.plot(dataTill2020.index[250:], dataTill2020.iloc[250:,2],label='Индекс', color='#6666ff', lw=1.5)   
ax.axvline(dataTill2020.index[sPEMAFirstOneIndex], color='#999999', linestyle='-', label='Инд. вкл.')
ax.axvline(dataTill2020.index[sPADFirstOneIndex], color='#999999', linestyle='-')
ax.axvline(dataTill2020.index[sPKAMAFirstOneIndex], color='#999999', linestyle='-')
ax.axvline(dataTill2020.index[sPMACDFirstOneIndex], color='#999999', linestyle='-')
ax.axvline(dataTill2020.index[sPTRIXFirstOneIndex], color='#999999', linestyle='-')
ax.axvline(dataTill2020.index[wholeSPHedgeFirstOneIndex], color='#00cc66', linestyle='-', label='Все вкл.')

ax.axvline(dataTill2020.index[sPADFirstZeroIndex], color='#999999', linestyle='--', label='Инд. выкл.')
ax.axvline(dataTill2020.index[sPMACDFirstZeroIndex], color='#999999', linestyle='--')

ax.legend(loc=0, fontsize=8)
ax.set_title('S&P500')           
ax.xaxis.set_major_locator(months)         

ax = fig.add_subplot(1,2,2)
ax.plot(dataTill2020.index[250:], dataTill2020.iloc[250:,5],label='Индекс', color='#6666ff', lw=1.5)   
ax.axvline(dataTill2020.index[rTSIEMAFirstOneIndex], color='#999999', linestyle='-', label='Инд. вкл.')
ax.axvline(dataTill2020.index[rTSIADFirstOneIndex], color='#999999', linestyle='-')
ax.axvline(dataTill2020.index[rTSIKAMAFirstOneIndex], color='#999999', linestyle='-')
ax.axvline(dataTill2020.index[rTSIMACDFirstOneIndex], color='#999999', linestyle='-')
ax.axvline(dataTill2020.index[rTSITRIXFirstOneIndex], color='#999999', linestyle='-')
ax.axvline(dataTill2020.index[wholeRTSIHedgeFirstOneIndex], color='#00cc66', linestyle='-', label='Все вкл.')

ax.axvline(dataTill2020.index[rTSIADFirstZeroIndex], color='#999999', linestyle='--', label='Инд. выкл.')

ax.set_title('RTSI')           
ax.xaxis.set_major_locator(months)
fig.autofmt_xdate()         

fig.savefig('./plots/new_dynamics.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)
'''

#%%% PLOT WITH SAMPLE DYNAMICS OF INDEX AND FUNDAMENTAL INDICATOR
'''
fig = plt.figure(figsize=(6.488, 2.81))
ax = fig.add_subplot()
ax.plot(closePriceDF.index[3800:6500], closePriceDF.iloc[3800:6500,1], label='S&P 500', color='#6666ff', lw=1.5)
ax.set_ylabel('Пункты S&P 500')
        
ax2 = ax.twinx()
ax2.plot(closePriceDF.index[3800:6500], indPB.iloc[3800:6500,1], label='Price/Book', color='#00cc66', lw=1.5)
ax2.set_ylim(2.5, 7)
ax2.set_ylabel('Индикатор P/B')

ax2.axvline(closePriceDF.index[4540], color='#999999', linestyle='-', label='Инд. вкл.')
ax2.axvline(closePriceDF.index[5520], color='#999999', linestyle='--', label='Инд. выкл.')
handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

fig.legend([handles[0], handles2[0], handles2[1], handles2[2]], [labels[0], labels2[0],labels2[1], labels2[2]], bbox_to_anchor=[0.28,0.865], fancybox=False, shadow=False,fontsize=8, framealpha=1)
fig.savefig('./plots/fundamental_sample.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)
'''
#%% PLOT WITH SAMPLE DYNAMICS OF INDEX AND TECHINICAL INDICATOR
'''
sPEmaShort = closePriceDF.iloc[:,1].ewm(span=10, adjust=False).mean()
sPEmaLong = closePriceDF.iloc[:,1].ewm(span=100, adjust=False).mean()
sPEmaSignal = EMA_sign(closePriceDF.iloc[:,1], 10, 100)

fig = plt.figure(figsize=(6.488, 2.81))
ax = fig.add_subplot()
ax.plot(closePriceDF.index[5200:6400], closePriceDF.iloc[5200:6400,1], label='S&P 500', color='#6666ff', lw=1.5)
ax.plot(closePriceDF.index[5200:6400], sPEmaShort[5200:6400], label='EMA кор.', color='#999999', lw=1.5)
ax.plot(closePriceDF.index[5200:6400], sPEmaLong[5200:6400], label='EMA длин.', color='#ff6699', lw=1.5)
ax.set_ylabel('Пункты S&P 500')

ax2 = ax.twinx()
ax2.plot(closePriceDF.index[5200:6400], sPEmaSignal[5200:6400], label='сигнал EMA', color='#00cc66', lw=1.5)
#ff6699
ax2.set_ylabel('Сигнал EMA')
ax2.set_yticklabels([])
ax2.set_yticks([])

handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

fig.legend([handles[0], handles[1], handles[2], handles2[0]], [labels[0], labels[1], labels[2], labels2[0]], bbox_to_anchor=[0.89,0.865], fancybox=False, shadow=False,fontsize=8, framealpha=1)
fig.savefig('./plots/technical_sample.jpg', dpi=300, quality=95, transparent=False, bbox_inches='tight', pad_inches=0.04, metadata=None)
'''


