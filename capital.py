import pandas as pd
import math
import numpy as np
from derivatives import futures_price, Option, CallOpt, PutOpt
from TA_FA import EMA_sign, AD_sign, KAMA_sign, MACD_sign, TRIX_sign
from days_to_exec import days_left, new_futures_dates, give_first_one, give_first_zero, period_to_hold_options, dates_tuple, get_condition_type1, get_condition_type2, get_condition_type3,get_condition_type4
#%%
def create_portfolio(P, daysToHoldOptions, daysToRenew, daysToSell, optionPrice):
     '''Function that creates portfolio dynamics for given data of futures price, options, and dates to buy and sell psition 
     '''
     capitalBH = []
     moneyInPortfolioToIndexRatio = []
     firstNotNull = pd.Series(P).notna().idxmax()
     for i in range(0,len(daysToHoldOptions)):
          if pd.isna(P[i]):
               capitalBH.append(float('nan'))
               moneyInPortfolioToIndexRatio.append(float('nan'))
          elif i==firstNotNull:
               if daysToHoldOptions[i]==0:
                    capitalBH.append(P[i])
                    moneyInPortfolioToIndexRatio.append(1)
               else:
                    moneyInPortfolioToIndexRatio.append(1)
                    capitalBH.append(P[i]-optionPrice[i])
          elif daysToHoldOptions[i]==0:
               if daysToSell[i]==1:
                    moneyInPortfolioToIndexRatio.append(moneyInPortfolioToIndexRatio[i-1])
                    capitalBH.append(capitalBH[i-1]*(P[i]/P[i-1])+optionPrice[i]*moneyInPortfolioToIndexRatio[i])
               else:
                    moneyInPortfolioToIndexRatio.append(moneyInPortfolioToIndexRatio[i-1])
                    capitalBH.append(capitalBH[i-1]*(P[i]/P[i-1]))
          else:
               if daysToRenew[i]==1:
                    moneyInPortfolioToIndexRatio.append(capitalBH[i-1]/P[i-1])
                    capitalBH.append(capitalBH[i-1]*(P[i]/P[i-1])-optionPrice[i]*moneyInPortfolioToIndexRatio[i])
               else:
                    moneyInPortfolioToIndexRatio.append(moneyInPortfolioToIndexRatio[i-1])
                    capitalBH.append(capitalBH[i-1]*(P[i]/P[i-1])) 
     totalCapital = [(x+y*z) if daysToSell[ind]!=1 else x for ind, x, y, z in zip(list(range(0, len(daysToSell))), capitalBH, optionPrice, moneyInPortfolioToIndexRatio)]
     return totalCapital
# think about comissions 

#%%
def mean_return(capitalDynamics):
     '''Provides annual return calculation.
     :capitalDynamics: list with capital over time period 
     :output: float, value of mean annual return
     '''
     portReturns = list(pd.Series(capitalDynamics).pct_change())
     annualTotalPortRateOfRet = (capitalDynamics[-1] / capitalDynamics[pd.Series(capitalDynamics).notna().idxmax()]) ** (252 / len(pd.Series(portReturns).dropna())) - 1
     return annualTotalPortRateOfRet 
#%%
def sharpe_ratio(capitalDynamics):
     '''Sharpe ratio calculation (annual return / annual volatility)
     :capitalDynamics: list with capital over time period (preferred format: fitst value equals 1)
     :output: float, value of Sharpe ratio
     '''
     portReturns = list(pd.Series(capitalDynamics).pct_change())
     annualTotalPortRateOfRet = (capitalDynamics[-1] / capitalDynamics[pd.Series(capitalDynamics).notna().idxmax()]) ** (252 / len(pd.Series(portReturns).dropna())) - 1
     annualPortStdDev = np.std(pd.Series(portReturns).dropna())*math.sqrt(252)
     return annualTotalPortRateOfRet/annualPortStdDev

#%%
def max_drawdown(capitalDyn):
     '''Calculation of maximum drawdown for some dynamics during observed period.
     :capitalDyn: list with capital dynamics (spot price, portfolio)
     :return: float - absolute value of maximum drawdown (L-H)/H.
     '''
     maximum = []
     maximum.append(capitalDyn[0]) 
     for i in range(1, len(capitalDyn)):
          if pd.isna(capitalDyn[i]): 
               maximum.append(float('nan'))
          elif pd.notna(capitalDyn[i]) and pd.isna(capitalDyn[i-1]):
               maximum.append(capitalDyn[i])
          elif capitalDyn[i]>=maximum[i-1]:
               maximum.append(capitalDyn[i])
          else:
               maximum.append(maximum[i-1])

     minimum = []
     minimum.append(capitalDyn[0]) 
     for i in range(1, len(capitalDyn)):
          if pd.isna(capitalDyn[i]): 
               minimum.append(float('nan'))
          elif pd.notna(capitalDyn[i]) and pd.isna(capitalDyn[i-1]):
               minimum.append(capitalDyn[i])
          elif capitalDyn[i]<=minimum[i-1]:
               minimum.append(capitalDyn[i])
          elif maximum[i]!=maximum[i-1]:
               minimum.append(capitalDyn[i])
          else:
               minimum.append(minimum[i-1])

     drawdowns = [(x-y)/y for x,y in zip(minimum, maximum)]
     biggestDrawdown = -1*np.nanmin(drawdowns)
     return biggestDrawdown
     
#%%
def calculate_strategies(countrynum, R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate):
     '''Gives pd.DF with time series for options and option strategies.
     :inputs: are the same as in Option class
     :output: DF with named strategies
     '''
     putATM = PutOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate)
     put95 = PutOpt(95, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate)
     put90 = PutOpt(90, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate)
     #put85 = PutOpt(85, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate)
     #put80 = PutOpt(80, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate)
     #straddle = [x+y for x,y in zip(putATM, CallOpt(100, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     strangle5 = [x+y for x,y in zip(put95, CallOpt(105, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     strangle10 = [x+y for x,y in zip(put90, CallOpt(110, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     #strangle15 = [x+y for x,y in zip(put85, CallOpt(115, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     #strangle20 = [x+y for x,y in zip(put80, CallOpt(120, 0.2, 'long', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     #bearSpread10090 = [x+y for x,y in zip(putATM, PutOpt(90, 0.2, 'short', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     bearSpread10085 = [x+y for x,y in zip(putATM, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     bearSpread10080 = [x+y for x,y in zip(putATM, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     bearSpread9585 = [x+y for x,y in zip(put95, PutOpt(85, 0.2, 'short', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     #bearSpread9580 = [x+y for x,y in zip(put95, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     #bearSpread9080 = [x+y for x,y in zip(put90, PutOpt(80, 0.2, 'short', countrynum).option_price(R, T, F, sigma, daysToRenew, daysToSell, daysToCalculate))]
     #optZip = list(zip(putATM, put95, put90, put85, put80, straddle, strangle5, strangle10, strangle15, strangle20, bearSpread10090, bearSpread10085, bearSpread10080, bearSpread9585, bearSpread9580, bearSpread9080))
     #optDF = pd.DataFrame(optZip, columns = ['putATM', 'put95', 'put90', 'put85', 'put80', 'straddle', 'strangle5', 'strangle10', 'strangle15', 'strangle20', 'bearSpread10090', 'bearSpread10085', 'bearSpread10080', 'bearSpread9585', 'bearSpread9580', 'bearSpread9080'])
     optZip = list(zip(putATM, put95, strangle5, strangle10, bearSpread10085, bearSpread10080, bearSpread9585))
     optDF = pd.DataFrame(optZip, columns = ['putATM', 'put95', 'strangle5', 'strangle10', 'bearSpread10085', 'bearSpread10080', 'bearSpread9585'])
     return optDF

#%%
def first_portfolios_with_given_options(spotPrice, daysToHold, daysToRenew, daysToSell, optionStrategies):
     '''Creates pd.DF of capital behaviour with given spot price and options DF (from calculate_strategies)
     :spotPrice: list of spot price
     :daysToHold: datesTuple[2] usually
     :daysToRenew: datesTuple[0]
     :daysToSell: datesTuple[1]
     :optionStrategies: pd.DF with options data
     :output: pd.DF with capital from the first options added (for the further adding another function is required)
     '''
     matrixOfPortfolios = np.empty((optionStrategies.shape[0], optionStrategies.shape[1]))
     for i in range(0,optionStrategies.shape[1]):
          matrixOfPortfolios[:,i] = create_portfolio(spotPrice, daysToHold, daysToRenew, daysToSell, list(optionStrategies.iloc[:,i]))
     return pd.DataFrame(matrixOfPortfolios)

#%%
def portfolios_with_given_options(previousPortf, daysToHold, daysToRenew, daysToSell, optionStrategiesDF):
     '''Creates pd.DF of capital behaviour with given previously created portfolios and options DF (from calculate_strategies)
     :previousPortf: pd.DF of previously created portfolios
     :daysToHold: datesTuple[2] usually
     :daysToRenew: datesTuple[0]
     :daysToSell: datesTuple[1]
     :optionStrategies: pd.DF with options data
     :output: pd.DF with capital from the first options added (for the further adding another function is required)
     '''
     matrixOfPortfolios = np.empty((optionStrategiesDF.shape[0], optionStrategiesDF.shape[1]))
     for i in range(0,optionStrategiesDF.shape[1]):
          matrixOfPortfolios[:,i] = create_portfolio(list(previousPortf.iloc[:,i]), daysToHold, daysToRenew, daysToSell, list(optionStrategiesDF.iloc[:,i]))
     return pd.DataFrame(matrixOfPortfolios)

#%%
def sharpe_ratios_vector(finalPortfolio):
     '''Gives pd.DF set with dimensions 1xnumber of strategies for given set of FA rules and TA periods (which is later added to big matrix, where the best strategy and periods are chosen)
     :finalPortfolio: when all the TA indicators are added
     :output: pd.DF with ratios for each strategy
     '''
     pBT1SharpeRatiosList = []
     for i in range(0, finalPortfolio.shape[1]):
          pBT1SharpeRatiosList.append(sharpe_ratio(list(finalPortfolio.iloc[:, i])))
     return pd.DataFrame(pBT1SharpeRatiosList)     
     