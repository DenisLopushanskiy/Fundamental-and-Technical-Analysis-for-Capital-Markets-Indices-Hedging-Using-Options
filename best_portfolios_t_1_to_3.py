import pandas as pd
import math
import numpy as np
from derivatives import futures_price, Option, CallOpt, PutOpt
from TA_FA import EMA_sign, AD_sign, KAMA_sign, MACD_sign, TRIX_sign
from days_to_exec import days_left, new_futures_dates, give_first_one, give_first_zero, period_to_hold_options, dates_tuple, get_condition_type1, get_condition_type2, get_condition_type3,get_condition_type4
from capital import create_portfolio, sharpe_ratio, calculate_strategies, first_portfolios_with_given_options, portfolios_with_given_options, sharpe_ratios_vector
#%%
def EMA_best_portfolios(listWithParams, clPrSeries, condFA, countrynum, R, T, F,  sigma,daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit):
     '''Messy function that calculates portfolios (for EMA these are the first portfolios) with B&H and options for each option strategy best with best parameters (calculated earlier in condition_type module) and takes list of best parameters as inputs.
     :listWithParams: list of lists with pairs of parameters (7 pairs). Each pair is later used to its own strategy (i.e. pair with index 0 is for putATM strategy)
     :clPrSeries: is used for TA calculation
     :condFA: type 1 condition of FA used in condition_types module
     :R, T, F,  sigma,daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit: used for options calculation
     :output: DF with portfolio dynamics for each strategy (later added AD, KAMA, MACD, TRIX options)
     '''
     daysToRenewMatrix = np.empty((len(clPrSeries), 7))
     daysToSellMatrix = np.empty((len(clPrSeries), 7))
     daysToHoldMatrix = np.empty((len(clPrSeries), 7))
     
     eMACond = EMA_sign(clPrSeries, int(listWithParams[0][0]), int(listWithParams[0][1]))
     fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,0] = datesTuple[0]
     daysToSellMatrix[:,0] = datesTuple[1]
     daysToHoldMatrix[:,0] = datesTuple[2]
     putATM = PutOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
     eMACond = EMA_sign(clPrSeries, int(listWithParams[1][0]), int(listWithParams[1][1]))
     fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,1] = datesTuple[0]
     daysToSellMatrix[:,1] = datesTuple[1]
     daysToHoldMatrix[:,1] = datesTuple[2]
     put95 = PutOpt(95, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
# =============================================================================
     eMACond = EMA_sign(clPrSeries, int(listWithParams[3][0]), int(listWithParams[3][1]))
     fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
#      datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,2] = datesTuple[0]
#      daysToSellMatrix[:,2] = datesTuple[1]
#      daysToHoldMatrix[:,2] = datesTuple[2]
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      eMACond = EMA_sign(clPrSeries, int(listWithParams[3][0]), int(listWithParams[3][1]))
#      fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
#      datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,3] = datesTuple[0]
#      daysToSellMatrix[:,3] = datesTuple[1]
#      daysToHoldMatrix[:,3] = datesTuple[2]
#      put85 = PutOpt(85, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      eMACond = EMA_sign(clPrSeries, int(listWithParams[4][0]), int(listWithParams[4][1]))
#      fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
#      datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,4] = datesTuple[0]
#      daysToSellMatrix[:,4] = datesTuple[1]
#      daysToHoldMatrix[:,4] = datesTuple[2]
#      put80 = PutOpt(80, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      eMACond = EMA_sign(clPrSeries, int(listWithParams[5][0]), int(listWithParams[5][1]))
#      fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
#      datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,5] = datesTuple[0]
#      daysToSellMatrix[:,5] = datesTuple[1]
#      daysToHoldMatrix[:,5] = datesTuple[2]
#      straddle = [x+y for x,y in zip(putATM, CallOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     eMACond = EMA_sign(clPrSeries, int(listWithParams[2][0]), int(listWithParams[2][1]))
     fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,2] = datesTuple[0]
     daysToSellMatrix[:,2] = datesTuple[1]
     daysToHoldMatrix[:,2] = datesTuple[2]
     strangle5 = [x+y for x,y in zip(put95, CallOpt(105, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     eMACond = EMA_sign(clPrSeries, int(listWithParams[3][0]), int(listWithParams[3][1]))
     fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,3] = datesTuple[0]
     daysToSellMatrix[:,3] = datesTuple[1]
     daysToHoldMatrix[:,3] = datesTuple[2]
     strangle10 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      eMACond = EMA_sign(clPrSeries, int(listWithParams[8][0]), int(listWithParams[8][1]))
#      fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
#      datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,8] = datesTuple[0]
#      daysToSellMatrix[:,8] = datesTuple[1]
#      daysToHoldMatrix[:,8] = datesTuple[2]
#      strangle15 = [x+y for x,y in zip(put85, CallOpt(115, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      eMACond = EMA_sign(clPrSeries, int(listWithParams[9][0]), int(listWithParams[9][1]))
#      fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
#      datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,9] = datesTuple[0]
#      daysToSellMatrix[:,9] = datesTuple[1]
#      daysToHoldMatrix[:,9] = datesTuple[2]
#      strangle20 = [x+y for x,y in zip(put80, CallOpt(120, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      eMACond = EMA_sign(clPrSeries, int(listWithParams[10][0]), int(listWithParams[10][1]))
#      fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
#      datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,10] = datesTuple[0]
#      daysToSellMatrix[:,10] = datesTuple[1]
#      daysToHoldMatrix[:,10] = datesTuple[2]
#      bearSpread10090 = [x+y for x,y in zip(putATM, PutOpt(90, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     eMACond = EMA_sign(clPrSeries, int(listWithParams[4][0]), int(listWithParams[4][1]))
     fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,4] = datesTuple[0]
     daysToSellMatrix[:,4] = datesTuple[1]
     daysToHoldMatrix[:,4] = datesTuple[2]
     bearSpread10085 = [x+y for x,y in zip(putATM, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     eMACond = EMA_sign(clPrSeries, int(listWithParams[5][0]), int(listWithParams[5][1]))
     fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,5] = datesTuple[0]
     daysToSellMatrix[:,5] = datesTuple[1]
     daysToHoldMatrix[:,5] = datesTuple[2]
     bearSpread10080 = [x+y for x,y in zip(putATM, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     eMACond = EMA_sign(clPrSeries, int(listWithParams[6][0]), int(listWithParams[6][1]))
     fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
     datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,6] = datesTuple[0]
     daysToSellMatrix[:,6] = datesTuple[1]
     daysToHoldMatrix[:,6] = datesTuple[2]
     bearSpread9585 = [x+y for x,y in zip(put95, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      eMACond = EMA_sign(clPrSeries, int(listWithParams[14][0]), int(listWithParams[14][1]))
#      fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
#      datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,14] = datesTuple[0]
#      daysToSellMatrix[:,14] = datesTuple[1]
#      daysToHoldMatrix[:,14] = datesTuple[2]
#      bearSpread9580 = [x+y for x,y in zip(put95, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      eMACond = EMA_sign(clPrSeries, int(listWithParams[15][0]), int(listWithParams[15][1]))
#      fAT1PlusEMA = [x*y for x, y in zip(condFA,eMACond)]
#      datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,15] = datesTuple[0]
#      daysToSellMatrix[:,15] = datesTuple[1]
#      daysToHoldMatrix[:,15] = datesTuple[2]
#      bearSpread9080 = [x+y for x,y in zip(put90, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     optZip = list(zip(putATM, put95, strangle5, strangle10,  bearSpread10085, bearSpread10080, bearSpread9585))
     optDF = pd.DataFrame(optZip)
     daysToRenewDF = pd.DataFrame(daysToRenewMatrix)
     daysToSellDF = pd.DataFrame(daysToSellMatrix)
     daysToHoldDF = pd.DataFrame(daysToHoldMatrix)
     eMABestPortfolios = np.empty((len(clPrSeries), 7))
     for n in range(0, eMABestPortfolios.shape[1]):
          eMABestPortfolios[:, n] = create_portfolio(list(clPrSeries), list(daysToHoldDF.iloc[:,n]), list(daysToRenewDF.iloc[:,n]), list(daysToSellDF.iloc[:,n]), list(optDF.iloc[:, n]))
     eMABestPortfoliosDF = pd.DataFrame(eMABestPortfolios)
     return eMABestPortfoliosDF

#%%
def AD_best_portfolios(listWithParams, hgPrSeries, lwPrSeries, condFA, countrynum, R, T, F,  sigma, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit, eMABestPortfoliosDF):
     '''Calculates portfolios with previously calculated portfolios (here from EMA condition) and options for each option strategy with best parameters (calculated earlier in condition_type module) and takes list of best parameters as inputs.
     :listWithParams: list of lists with pairs of parameters (7 pairs). Each pair is later used to its own strategy (i.e. pair with index 0 is for putATM strategy)
     :hgPrSeries, lwPrSeries,: are used for TA calculation
     :condFA: type 1 condition of FA used in condition_types module
     :R, T, F,  sigma,daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit: used for options calculation
     :output: DF with portfolio dynamics for each strategy (later added to KAMA, MACD, TRIX options)
     '''
     daysToRenewMatrix = np.empty((len(hgPrSeries), 7))
     daysToSellMatrix = np.empty((len(hgPrSeries), 7))
     daysToHoldMatrix = np.empty((len(hgPrSeries), 7))
     
     aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[0]))
     fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,0] = datesTuple[0]
     daysToSellMatrix[:,0] = datesTuple[1]
     daysToHoldMatrix[:,0] = datesTuple[2]
     putATM = PutOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
     aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[1]))
     fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,1] = datesTuple[0]
     daysToSellMatrix[:,1] = datesTuple[1]
     daysToHoldMatrix[:,1] = datesTuple[2]
     put95 = PutOpt(95, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
# =============================================================================
     aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[3]))
     fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
#      datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,2] = datesTuple[0]
#      daysToSellMatrix[:,2] = datesTuple[1]
#      daysToHoldMatrix[:,2] = datesTuple[2]
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[3]))
#      fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
#      datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,3] = datesTuple[0]
#      daysToSellMatrix[:,3] = datesTuple[1]
#      daysToHoldMatrix[:,3] = datesTuple[2]
#      put85 = PutOpt(85, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[4]))
#      fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
#      datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,4] = datesTuple[0]
#      daysToSellMatrix[:,4] = datesTuple[1]
#      daysToHoldMatrix[:,4] = datesTuple[2]
#      put80 = PutOpt(80, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[5]))
#      fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
#      datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,5] = datesTuple[0]
#      daysToSellMatrix[:,5] = datesTuple[1]
#      daysToHoldMatrix[:,5] = datesTuple[2]
#      straddle = [x+y for x,y in zip(putATM, CallOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[2]))
     fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,2] = datesTuple[0]
     daysToSellMatrix[:,2] = datesTuple[1]
     daysToHoldMatrix[:,2] = datesTuple[2]
     strangle5 = [x+y for x,y in zip(put95, CallOpt(105, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[3]))
     fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,3] = datesTuple[0]
     daysToSellMatrix[:,3] = datesTuple[1]
     daysToHoldMatrix[:,3] = datesTuple[2]
     strangle10 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[8]))
#      fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
#      datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,8] = datesTuple[0]
#      daysToSellMatrix[:,8] = datesTuple[1]
#      daysToHoldMatrix[:,8] = datesTuple[2]
#      strangle15 = [x+y for x,y in zip(put85, CallOpt(115, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[9]))
#      fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
#      datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,9] = datesTuple[0]
#      daysToSellMatrix[:,9] = datesTuple[1]
#      daysToHoldMatrix[:,9] = datesTuple[2]
#      strangle20 = [x+y for x,y in zip(put80, CallOpt(120, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[10]))
#      fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
#      datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,10] = datesTuple[0]
#      daysToSellMatrix[:,10] = datesTuple[1]
#      daysToHoldMatrix[:,10] = datesTuple[2]
#      bearSpread10090 = [x+y for x,y in zip(putATM, PutOpt(90, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[4]))
     fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,4] = datesTuple[0]
     daysToSellMatrix[:,4] = datesTuple[1]
     daysToHoldMatrix[:,4] = datesTuple[2]
     bearSpread10085 = [x+y for x,y in zip(putATM, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[5]))
     fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,5] = datesTuple[0]
     daysToSellMatrix[:,5] = datesTuple[1]
     daysToHoldMatrix[:,5] = datesTuple[2]
     bearSpread10080 = [x+y for x,y in zip(putATM, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[6]))
     fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
     datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,6] = datesTuple[0]
     daysToSellMatrix[:,6] = datesTuple[1]
     daysToHoldMatrix[:,6] = datesTuple[2]
     bearSpread9585 = [x+y for x,y in zip(put95, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[14]))
#      fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
#      datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,14] = datesTuple[0]
#      daysToSellMatrix[:,14] = datesTuple[1]
#      daysToHoldMatrix[:,14] = datesTuple[2]
#      bearSpread9580 = [x+y for x,y in zip(put95, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      aDCond = AD_sign(hgPrSeries, lwPrSeries, 5, 34, int(listWithParams[15]))
#      fAT1PlusAD = [x*y for x, y in zip(condFA,aDCond)]
#      datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,15] = datesTuple[0]
#      daysToSellMatrix[:,15] = datesTuple[1]
#      daysToHoldMatrix[:,15] = datesTuple[2]
#      bearSpread9080 = [x+y for x,y in zip(put90, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     optZip = list(zip(putATM, put95, strangle5, strangle10,  bearSpread10085, bearSpread10080, bearSpread9585))
     optDF = pd.DataFrame(optZip)
     daysToRenewDF = pd.DataFrame(daysToRenewMatrix)
     daysToSellDF = pd.DataFrame(daysToSellMatrix)
     daysToHoldDF = pd.DataFrame(daysToHoldMatrix)
     aDBestPortfolios = np.empty((len(hgPrSeries), 7))
     for n in range(0, aDBestPortfolios.shape[1]):
          aDBestPortfolios[:, n] = create_portfolio(list(eMABestPortfoliosDF.iloc[:,n]), list(daysToHoldDF.iloc[:,n]), list(daysToRenewDF.iloc[:,n]), list(daysToSellDF.iloc[:,n]), list(optDF.iloc[:, n]))
     aDBestPortfoliosDF = pd.DataFrame(aDBestPortfolios)
     return aDBestPortfoliosDF

#%%
def KAMA_best_portfolios(listWithParams, clPrSeries,  condFA, countrynum, R, T, F,  sigma, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit, aDBestPortfoliosDF):
     '''Calculates portfolios with previously calculated portfolios (here from EMA condition) and options for each option strategy with best parameters (calculated earlier in condition_type module) and takes list of best parameters as inputs.
     :listWithParams: list of lists with pairs of parameters (7 pairs). Each pair is later used to its own strategy (i.e. pair with index 0 is for putATM strategy)
     :clPrSeries: is used for TA calculation
     :condFA: type 1 condition of FA used in condition_types module
     :R, T, F,  sigma,daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit: used for options calculation
     :output: DF with portfolio dynamics for each strategy (later added to MACD, TRIX options)
     '''
     daysToRenewMatrix = np.empty((len(clPrSeries), 7))
     daysToSellMatrix = np.empty((len(clPrSeries), 7))
     daysToHoldMatrix = np.empty((len(clPrSeries), 7))
     
     kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[0]))
     fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,0] = datesTuple[0]
     daysToSellMatrix[:,0] = datesTuple[1]
     daysToHoldMatrix[:,0] = datesTuple[2]
     putATM = PutOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
     kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[1]))
     fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,1] = datesTuple[0]
     daysToSellMatrix[:,1] = datesTuple[1]
     daysToHoldMatrix[:,1] = datesTuple[2]
     put95 = PutOpt(95, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
# =============================================================================
     kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[3]))
     fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
#      datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,2] = datesTuple[0]
#      daysToSellMatrix[:,2] = datesTuple[1]
#      daysToHoldMatrix[:,2] = datesTuple[2]
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[3]))
#      fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
#      datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,3] = datesTuple[0]
#      daysToSellMatrix[:,3] = datesTuple[1]
#      daysToHoldMatrix[:,3] = datesTuple[2]
#      put85 = PutOpt(85, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[4]))
#      fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
#      datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,4] = datesTuple[0]
#      daysToSellMatrix[:,4] = datesTuple[1]
#      daysToHoldMatrix[:,4] = datesTuple[2]
#      put80 = PutOpt(80, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[5]))
#      fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
#      datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,5] = datesTuple[0]
#      daysToSellMatrix[:,5] = datesTuple[1]
#      daysToHoldMatrix[:,5] = datesTuple[2]
#      straddle = [x+y for x,y in zip(putATM, CallOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[2]))
     fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,2] = datesTuple[0]
     daysToSellMatrix[:,2] = datesTuple[1]
     daysToHoldMatrix[:,2] = datesTuple[2]
     strangle5 = [x+y for x,y in zip(put95, CallOpt(105, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[3]))
     fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,3] = datesTuple[0]
     daysToSellMatrix[:,3] = datesTuple[1]
     daysToHoldMatrix[:,3] = datesTuple[2]
     strangle10 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[8]))
#      fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
#      datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,8] = datesTuple[0]
#      daysToSellMatrix[:,8] = datesTuple[1]
#      daysToHoldMatrix[:,8] = datesTuple[2]
#      strangle15 = [x+y for x,y in zip(put85, CallOpt(115, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[9]))
#      fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
#      datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,9] = datesTuple[0]
#      daysToSellMatrix[:,9] = datesTuple[1]
#      daysToHoldMatrix[:,9] = datesTuple[2]
#      strangle20 = [x+y for x,y in zip(put80, CallOpt(120, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[10]))
#      fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
#      datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,10] = datesTuple[0]
#      daysToSellMatrix[:,10] = datesTuple[1]
#      daysToHoldMatrix[:,10] = datesTuple[2]
#      bearSpread10090 = [x+y for x,y in zip(putATM, PutOpt(90, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[4]))
     fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,4] = datesTuple[0]
     daysToSellMatrix[:,4] = datesTuple[1]
     daysToHoldMatrix[:,4] = datesTuple[2]
     bearSpread10085 = [x+y for x,y in zip(putATM, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[5]))
     fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,5] = datesTuple[0]
     daysToSellMatrix[:,5] = datesTuple[1]
     daysToHoldMatrix[:,5] = datesTuple[2]
     bearSpread10080 = [x+y for x,y in zip(putATM, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[6]))
     fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
     datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,6] = datesTuple[0]
     daysToSellMatrix[:,6] = datesTuple[1]
     daysToHoldMatrix[:,6] = datesTuple[2]
     bearSpread9585 = [x+y for x,y in zip(put95, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[14]))
#      fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
#      datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,14] = datesTuple[0]
#      daysToSellMatrix[:,14] = datesTuple[1]
#      daysToHoldMatrix[:,14] = datesTuple[2]
#      bearSpread9580 = [x+y for x,y in zip(put95, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      kAMACond = KAMA_sign(clPrSeries, 10, 2, 30, int(listWithParams[15]))
#      fAT1PlusKAMA = [x*y for x, y in zip(condFA,kAMACond)]
#      datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,15] = datesTuple[0]
#      daysToSellMatrix[:,15] = datesTuple[1]
#      daysToHoldMatrix[:,15] = datesTuple[2]
#      bearSpread9080 = [x+y for x,y in zip(put90, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     optZip = list(zip(putATM, put95, strangle5, strangle10,  bearSpread10085, bearSpread10080, bearSpread9585))
     optDF = pd.DataFrame(optZip)
     daysToRenewDF = pd.DataFrame(daysToRenewMatrix)
     daysToSellDF = pd.DataFrame(daysToSellMatrix)
     daysToHoldDF = pd.DataFrame(daysToHoldMatrix)
     kAMABestPortfolios = np.empty((len(clPrSeries), 7))
     for n in range(0, kAMABestPortfolios.shape[1]):
          kAMABestPortfolios[:, n] = create_portfolio(list(aDBestPortfoliosDF.iloc[:,n]), list(daysToHoldDF.iloc[:,n]), list(daysToRenewDF.iloc[:,n]), list(daysToSellDF.iloc[:,n]), list(optDF.iloc[:, n]))
     kAMABestPortfoliosDF = pd.DataFrame(kAMABestPortfolios)
     return kAMABestPortfoliosDF

#%%
def MACD_best_portfolios(listWithParams, clPrSeries,  condFA, countrynum, R, T, F,  sigma, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit, kAMABestPortfoliosDF):
     '''Calculates portfolios with previously calculated portfolios (here from EMA condition) and options for each option strategy with best parameters (calculated earlier in condition_type module) and takes list of best parameters as inputs.
     :listWithParams: list of lists with pairs of parameters (7 pairs). Each pair is later used to its own strategy (i.e. pair with index 0 is for putATM strategy)
     :clPrSeries: is used for TA calculation
     :condFA: type 1 condition of FA used in condition_types module
     :R, T, F,  sigma,daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit: used for options calculation
     :output: DF with portfolio dynamics for each strategy (later added to TRIX options)
     '''
     daysToRenewMatrix = np.empty((len(clPrSeries), 7))
     daysToSellMatrix = np.empty((len(clPrSeries), 7))
     daysToHoldMatrix = np.empty((len(clPrSeries), 7))
     
     mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[0]))
     fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,0] = datesTuple[0]
     daysToSellMatrix[:,0] = datesTuple[1]
     daysToHoldMatrix[:,0] = datesTuple[2]
     putATM = PutOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
     mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[1]))
     fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,1] = datesTuple[0]
     daysToSellMatrix[:,1] = datesTuple[1]
     daysToHoldMatrix[:,1] = datesTuple[2]
     put95 = PutOpt(95, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
# =============================================================================
     mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[3]))
     fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
#      datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,2] = datesTuple[0]
#      daysToSellMatrix[:,2] = datesTuple[1]
#      daysToHoldMatrix[:,2] = datesTuple[2]
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[3]))
#      fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
#      datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,3] = datesTuple[0]
#      daysToSellMatrix[:,3] = datesTuple[1]
#      daysToHoldMatrix[:,3] = datesTuple[2]
#      put85 = PutOpt(85, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[4]))
#      fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
#      datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,4] = datesTuple[0]
#      daysToSellMatrix[:,4] = datesTuple[1]
#      daysToHoldMatrix[:,4] = datesTuple[2]
#      put80 = PutOpt(80, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[5]))
#      fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
#      datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,5] = datesTuple[0]
#      daysToSellMatrix[:,5] = datesTuple[1]
#      daysToHoldMatrix[:,5] = datesTuple[2]
#      straddle = [x+y for x,y in zip(putATM, CallOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[2]))
     fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,2] = datesTuple[0]
     daysToSellMatrix[:,2] = datesTuple[1]
     daysToHoldMatrix[:,2] = datesTuple[2]
     strangle5 = [x+y for x,y in zip(put95, CallOpt(105, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[3]))
     fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,3] = datesTuple[0]
     daysToSellMatrix[:,3] = datesTuple[1]
     daysToHoldMatrix[:,3] = datesTuple[2]
     strangle10 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[8]))
#      fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
#      datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,8] = datesTuple[0]
#      daysToSellMatrix[:,8] = datesTuple[1]
#      daysToHoldMatrix[:,8] = datesTuple[2]
#      strangle15 = [x+y for x,y in zip(put85, CallOpt(115, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[9]))
#      fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
#      datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,9] = datesTuple[0]
#      daysToSellMatrix[:,9] = datesTuple[1]
#      daysToHoldMatrix[:,9] = datesTuple[2]
#      strangle20 = [x+y for x,y in zip(put80, CallOpt(120, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[10]))
#      fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
#      datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,10] = datesTuple[0]
#      daysToSellMatrix[:,10] = datesTuple[1]
#      daysToHoldMatrix[:,10] = datesTuple[2]
#      bearSpread10090 = [x+y for x,y in zip(putATM, PutOpt(90, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[4]))
     fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,4] = datesTuple[0]
     daysToSellMatrix[:,4] = datesTuple[1]
     daysToHoldMatrix[:,4] = datesTuple[2]
     bearSpread10085 = [x+y for x,y in zip(putATM, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[5]))
     fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,5] = datesTuple[0]
     daysToSellMatrix[:,5] = datesTuple[1]
     daysToHoldMatrix[:,5] = datesTuple[2]
     bearSpread10080 = [x+y for x,y in zip(putATM, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[6]))
     fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
     datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,6] = datesTuple[0]
     daysToSellMatrix[:,6] = datesTuple[1]
     daysToHoldMatrix[:,6] = datesTuple[2]
     bearSpread9585 = [x+y for x,y in zip(put95, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[14]))
#      fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
#      datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,14] = datesTuple[0]
#      daysToSellMatrix[:,14] = datesTuple[1]
#      daysToHoldMatrix[:,14] = datesTuple[2]
#      bearSpread9580 = [x+y for x,y in zip(put95, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      mACDCond = MACD_sign(clPrSeries, 26, 12, int(listWithParams[15]))
#      fAT1PlusMACD = [x*y for x, y in zip(condFA,mACDCond)]
#      datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,15] = datesTuple[0]
#      daysToSellMatrix[:,15] = datesTuple[1]
#      daysToHoldMatrix[:,15] = datesTuple[2]
#      bearSpread9080 = [x+y for x,y in zip(put90, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     optZip = list(zip(putATM, put95, strangle5, strangle10,  bearSpread10085, bearSpread10080, bearSpread9585))
     optDF = pd.DataFrame(optZip)
     daysToRenewDF = pd.DataFrame(daysToRenewMatrix)
     daysToSellDF = pd.DataFrame(daysToSellMatrix)
     daysToHoldDF = pd.DataFrame(daysToHoldMatrix)
     mACDBestPortfolios = np.empty((len(clPrSeries), 7))
     for n in range(0, mACDBestPortfolios.shape[1]):
          mACDBestPortfolios[:, n] = create_portfolio(list(kAMABestPortfoliosDF.iloc[:,n]), list(daysToHoldDF.iloc[:,n]), list(daysToRenewDF.iloc[:,n]), list(daysToSellDF.iloc[:,n]), list(optDF.iloc[:, n]))
     mACDBestPortfoliosDF = pd.DataFrame(mACDBestPortfolios)
     return mACDBestPortfoliosDF

#%%
def TRIX_best_portfolios(listWithParams, clPrSeries,  condFA, countrynum, R, T, F,  sigma, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit, mACDBestPortfoliosDF):
     '''Calculates portfolios with previously calculated portfolios (here from EMA condition) and options for each option strategy with best parameters (calculated earlier in condition_type module) and takes list of best parameters as inputs.
     :listWithParams: list of lists with pairs of parameters (7 pairs). Each pair is later used to its own strategy (i.e. pair with index 0 is for putATM strategy)
     :clPrSeries: is used for TA calculation
     :condFA: type 1 condition of FA used in condition_types module
     :R, T, F,  sigma,daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit: used for options calculation
     :output: DF with portfolio dynamics for each strategy. This is the final set. Later in condition_types Sharpe coefficients are calculated for TRIX portfolios
     '''
     daysToRenewMatrix = np.empty((len(clPrSeries), 7))
     daysToSellMatrix = np.empty((len(clPrSeries), 7))
     daysToHoldMatrix = np.empty((len(clPrSeries), 7))
     
     tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[0]))
     fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,0] = datesTuple[0]
     daysToSellMatrix[:,0] = datesTuple[1]
     daysToHoldMatrix[:,0] = datesTuple[2]
     putATM = PutOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
     tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[1]))
     fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,1] = datesTuple[0]
     daysToSellMatrix[:,1] = datesTuple[1]
     daysToHoldMatrix[:,1] = datesTuple[2]
     put95 = PutOpt(95, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
     
# =============================================================================
     tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[3]))
     fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
#      datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,2] = datesTuple[0]
#      daysToSellMatrix[:,2] = datesTuple[1]
#      daysToHoldMatrix[:,2] = datesTuple[2]
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[3]))
#      fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
#      datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,3] = datesTuple[0]
#      daysToSellMatrix[:,3] = datesTuple[1]
#      daysToHoldMatrix[:,3] = datesTuple[2]
#      put85 = PutOpt(85, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[4]))
#      fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
#      datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,4] = datesTuple[0]
#      daysToSellMatrix[:,4] = datesTuple[1]
#      daysToHoldMatrix[:,4] = datesTuple[2]
#      put80 = PutOpt(80, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3])
#      
#      tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[5]))
#      fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
#      datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,5] = datesTuple[0]
#      daysToSellMatrix[:,5] = datesTuple[1]
#      daysToHoldMatrix[:,5] = datesTuple[2]
#      straddle = [x+y for x,y in zip(putATM, CallOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
# =============================================================================
     tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[2]))
     fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,2] = datesTuple[0]
     daysToSellMatrix[:,2] = datesTuple[1]
     daysToHoldMatrix[:,2] = datesTuple[2]
     strangle5 = [x+y for x,y in zip(put95, CallOpt(105, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[3]))
     fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,3] = datesTuple[0]
     daysToSellMatrix[:,3] = datesTuple[1]
     daysToHoldMatrix[:,3] = datesTuple[2]
     strangle10 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[8]))
#      fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
#      datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,8] = datesTuple[0]
#      daysToSellMatrix[:,8] = datesTuple[1]
#      daysToHoldMatrix[:,8] = datesTuple[2]
#      strangle15 = [x+y for x,y in zip(put85, CallOpt(115, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[9]))
#      fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
#      datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,9] = datesTuple[0]
#      daysToSellMatrix[:,9] = datesTuple[1]
#      daysToHoldMatrix[:,9] = datesTuple[2]
#      strangle20 = [x+y for x,y in zip(put80, CallOpt(120, 0.2, 'long', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[10]))
#      fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
#      datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,10] = datesTuple[0]
#      daysToSellMatrix[:,10] = datesTuple[1]
#      daysToHoldMatrix[:,10] = datesTuple[2]
#      bearSpread10090 = [x+y for x,y in zip(putATM, PutOpt(90, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[4]))
     fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,4] = datesTuple[0]
     daysToSellMatrix[:,4] = datesTuple[1]
     daysToHoldMatrix[:,4] = datesTuple[2]
     bearSpread10085 = [x+y for x,y in zip(putATM, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[5]))
     fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,5] = datesTuple[0]
     daysToSellMatrix[:,5] = datesTuple[1]
     daysToHoldMatrix[:,5] = datesTuple[2]
     bearSpread10080 = [x+y for x,y in zip(putATM, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
     tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[6]))
     fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
     datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
     daysToRenewMatrix[:,6] = datesTuple[0]
     daysToSellMatrix[:,6] = datesTuple[1]
     daysToHoldMatrix[:,6] = datesTuple[2]
     bearSpread9585 = [x+y for x,y in zip(put95, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
     
# =============================================================================
#      tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[14]))
#      fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
#      datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,14] = datesTuple[0]
#      daysToSellMatrix[:,14] = datesTuple[1]
#      daysToHoldMatrix[:,14] = datesTuple[2]
#      bearSpread9580 = [x+y for x,y in zip(put95, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
#      
#      tRIXCond = TRIX_sign(clPrSeries, int(listWithParams[15]))
#      fAT1PlusTRIX = [x*y for x, y in zip(condFA,tRIXCond)]
#      datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)
#      daysToRenewMatrix[:,15] = datesTuple[0]
#      daysToSellMatrix[:,15] = datesTuple[1]
#      daysToHoldMatrix[:,15] = datesTuple[2]
#      bearSpread9080 = [x+y for x,y in zip(put90, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, datesTuple[0], datesTuple[1], datesTuple[3]))]
# =============================================================================
     
     optZip = list(zip(putATM, put95, strangle5, strangle10,  bearSpread10085, bearSpread10080, bearSpread9585))
     optDF = pd.DataFrame(optZip)
     daysToRenewDF = pd.DataFrame(daysToRenewMatrix)
     daysToSellDF = pd.DataFrame(daysToSellMatrix)
     daysToHoldDF = pd.DataFrame(daysToHoldMatrix)
     tRIXBestPortfolios = np.empty((len(clPrSeries), 7))
     for n in range(0, tRIXBestPortfolios.shape[1]):
          tRIXBestPortfolios[:, n] = create_portfolio(list(mACDBestPortfoliosDF.iloc[:,n]), list(daysToHoldDF.iloc[:,n]), list(daysToRenewDF.iloc[:,n]), list(daysToSellDF.iloc[:,n]), list(optDF.iloc[:, n]))
     tRIXBestPortfoliosDF = pd.DataFrame(tRIXBestPortfolios)
     return tRIXBestPortfoliosDF