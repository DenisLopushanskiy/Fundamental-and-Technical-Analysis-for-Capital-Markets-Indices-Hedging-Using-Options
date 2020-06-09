import pandas as pd
import math
from scipy import stats
#%%
def futures_price(interpRates, daysToExer, spotPrice):
     '''Future price for given index is calculated.
     '''
     futuresPrice = [a * math.exp(math.log(b/100+1)*c/252) for a, b, c in zip(spotPrice, interpRates, daysToExer)]
     return futuresPrice

#%% class option is defined (able to see strikes list for a given strike percent)
class Option:
     strikes = [80,85,90,95,100,105,110,115,120]
     volSurface = [1.6,1.4,1.3,1.1,1,1,1,1.05,1.15]
     spreads = [[0.1,0.05,0.02,0.01,0.01,0.01,0.02,0.05,0.1],
                [0.15,0.05,0.02,0.01,0.01,0.03,0.1,0.2,0.25]]
     '''Class that creates  options and gives d1 and d2 for them. Constructor takes strike percents as an integer (i.e. 100 for ATM) and the shave of capital to hedge (i.e. 0.2 or 1)
     '''
     def __init__(self, strike_percent, hedge_percent, position_type, countrynum):
          self.strike_percent = strike_percent #for instance, 100 means ATM option
          self.hedge_percent = hedge_percent #example: 1 means to hedge all the portfolio
          self.position_type = position_type #long or short
          self.countrynum = countrynum #to determine spreads
     
     def show_option_strikes(self, daysForOptionCalculation, daysToRenew, F):
          '''Shows strikes for a given strike percents 
          :daysForOptionCalculation: list of 1 and 0 that shows days when position is hold + the sell date (for option calculation)
          :daysToRenew: list of 1 and 0 when to reconsider position (after expiration and when the system says initially to buy options)
          :F: list of futures prices for a given index
          :return: list with strikes and NaN when position is not opened
          '''
          strike_points = [x * self.strike_percent / 100 if daysToRenew[ind]==1 and daysForOptionCalculation[ind]==1 else ('substitute' if daysToRenew[ind]==0 and daysForOptionCalculation[ind]==1 else float('nan')) for ind, x in enumerate(F)] #substitute means that those values should be the same as on the position renew date (buy ATM with some strike I have the strike unchanged till the expiration)
          strike = [float('nan')]*len(strike_points)
          for i in range(0, len(strike)):
               if i==0:
                    if pd.isna(strike_points[i]):
                         strike[i] = float('nan')
               else:
                    if strike_points[i]=='substitute':
                         strike[i] = strike[i-1]
                    else:
                         strike[i] = strike_points[i]
          return strike
                                        
     def d1_calculation(self, F, sigma, T, daysToRenew, daysForOptionCalculation):
          '''Calculates d1 according to Black formula for given dates determined by system
          :daysToRenew, F, daysForOptionCalculation: look at show_option_strikes
          :sigma: implied volatility given in percents (e.g. 20.3)
          :T: days left till expiration
          :return: list
          '''
          volatMultipl = self.volSurface[self.strikes.index(self.strike_percent)]
          K = self.show_option_strikes(daysForOptionCalculation, daysToRenew, F)
          d1 = [(math.log(f/k) + (s*volatMultipl/100)**2 / 2 * (t/252)) / (s*volatMultipl/100) / math.sqrt(t/252) if not T[ind]==0 and pd.notna(sigma[ind]) else (10000 if (pd.notna(sigma[ind]) and pd.notna(K[ind])) else float('nan')) for ind, f, k, s, t in zip(list(range(0,len(F))), F, K, sigma, T)] #value 10000 should be accounted when the price of option is calculated 
          return d1, K
     
     def d2_calculation(self, F, sigma, T, daysToRenew, daysForOptionCalculation):
         '''d2 calculation. Inputs and output type are the same as in d1.
         ''' 
         volatMultipl = self.volSurface[self.strikes.index(self.strike_percent)]
         d1, K = self.d1_calculation(F, sigma, T, daysToRenew, daysForOptionCalculation)
         d2 = [x - (s*volatMultipl/100) * math.sqrt(t/252) if not d1[ind]==10000 and pd.notna(d1[ind]) else (10000 if pd.notna(d1[ind]) else float('nan')) for ind, x, s, t in zip(list(range(0,len(d1))), d1, sigma, T)] #evalue 10000 should be accounted when the price of option is calculated
         return d2

#%%
class CallOpt(Option):
     '''
     Call price calculation. Inherits d1 and d2, price calculation is added. Constructor takes strike percents as an integer (i.e. 100 for ATM)
     '''     
     def __init__(self, strike_percent, hedge_percent, position_type, countrynum):
          Option.__init__(self, strike_percent, hedge_percent, position_type, countrynum) #for instance, 100 means ATM option
          
     def option_price(self, R, T, F, sigma, daysToRenew, daysToSell, daysForOptionCalculation):
          '''Call option price calculation.
          :R: list with interpolated rates
          :T, sigma: look at d1_alculation
          :daysToRenew, F, daysForOptionCalculation: look at show_option_strikes
          :return: list of prices for given date and strike percent with empty periods filled with zeros
          '''
          #K = self.show_option_strikes(daysForOptionCalculation, daysToRenew, F)
          D1, K = self.d1_calculation(F, sigma, T, daysToRenew, daysForOptionCalculation)
          D2 = self.d2_calculation(F, sigma, T, daysToRenew, daysForOptionCalculation)
          nD1 = stats.norm.cdf(D1)
          nD2 = stats.norm.cdf(D2)
          firstObserv = pd.Series(sigma).notna().idxmax() #how many before are empty
          nANList = [float('nan')]*firstObserv #empty list
          D1tr = D1[firstObserv:]
          #D2tr = D2[firstObserv:]
          nD1tr = nD1[firstObserv:]
          nD2tr = nD2[firstObserv:]
          Ktr = K[firstObserv:]
          Ftr = F[firstObserv:]
          Ttr = T[firstObserv:]
          Rtr = R[firstObserv:]
          #sigmatr = sigma[firstObserv:]
          #daysToRenewtr = daysToRenew[firstObserv:]
          #daysToSelltr = daysToSell[firstObserv:]
          #daysForOptionCalculationtr = daysForOptionCalculation[firstObserv:]
          spreadMultipl = self.spreads[0 if self.countrynum<=9 else 1][self.strikes.index(self.strike_percent)]
          callPrice = [math.exp(-math.log(r/100+1)*t/252) * (f*self.hedge_percent*nd1 - k*self.hedge_percent*nd2) if not (D1tr[ind]==10000 or pd.isna(Ktr[ind])) else (0 if (f - k) < 0 else (f - k)*self.hedge_percent)  for ind, r, t, f, k, nd1, nd2, in zip(list(range(0,len(Rtr))), Rtr, Ttr, Ftr, Ktr, nD1tr, nD2tr)]
          callPriceFull = nANList + callPrice
          callPriceNAFill = list(pd.Series(callPriceFull).fillna(value=0))
          if self.position_type=='long':
               callPriceWithSpreads = [callPriceNAFill[ind]*(1+spreadMultipl) if daysToRenew[ind]==1 else (callPriceNAFill[ind]*(1-spreadMultipl) if daysToSell[ind]==1 and D1[ind]!=10000 else callPriceNAFill[ind]) for ind, x in enumerate(callPriceNAFill)]
          else:
               callPriceWithSpreads = [-callPriceNAFill[ind]*(1-spreadMultipl) if daysToRenew[ind]==1 else (-callPriceNAFill[ind]*(1+spreadMultipl) if daysToSell[ind]==1 and D1[ind]!=10000 else -callPriceNAFill[ind]) for ind, x in enumerate(callPriceNAFill)]
          return callPriceWithSpreads
     
     def is_in_money(self, daysToRenew, F, daysForOptionCalculation):
          '''Shows periods when the option is in money
          '''
          K = self.show_option_strikes(daysForOptionCalculation, daysToRenew, F)
          moneyness = [1 if F[ind]>=K[ind] else (float('nan') if pd.isna(K[ind]) else 0) for ind, x, y in zip(list(range(0,len(F))), F, K)]
          return moneyness

#%%    
class PutOpt(Option):
     '''
     Put price calculation. Inherits d1 and d2, price calculation is added. Constructor takes strike percents as an integer (i.e. 100 for ATM)
     '''     
     def __init__(self, strike_percent, hedge_percent, position_type, countrynum):
          Option.__init__(self, strike_percent, hedge_percent, position_type, countrynum) #for instance, 100 means ATM option  
     
     def option_price(self, R, T, F, sigma, daysToRenew, daysToSell, daysForOptionCalculation):
          '''Put option price calculation. Inputs and output type are the same as in CallOpt
          '''
          #K = self.show_option_strikes(daysForOptionCalculation, daysToRenew, F)
          D1, K = self.d1_calculation(F, sigma, T, daysToRenew, daysForOptionCalculation)
          D2 = self.d2_calculation(F, sigma, T, daysToRenew, daysForOptionCalculation)
          minusD1 = [-1*x for x in D1]
          minusD2 = [-1*x for x in D2]
          nMinusD1 = stats.norm.cdf(minusD1)
          nMinusD2 = stats.norm.cdf(minusD2)
          firstObserv = pd.Series(sigma).notna().idxmax() #how many before are empty
          nANList = [float('nan')]*firstObserv #empty list
          D1tr = D1[firstObserv:]
          #D2tr = D2[firstObserv:]
          nMinusD1tr = nMinusD1[firstObserv:]
          nMinusD2tr = nMinusD2[firstObserv:]
          Ktr = K[firstObserv:]
          Ftr = F[firstObserv:]
          Ttr = T[firstObserv:]
          Rtr = R[firstObserv:]
          #sigmatr = sigma[firstObserv:]
          #daysToRenewtr = daysToRenew[firstObserv:]
          #daysToSelltr = daysToSell[firstObserv:]
          #daysForOptionCalculationtr = daysForOptionCalculation[firstObserv:]
          spreadMultipl = self.spreads[0 if self.countrynum<=9 else 1][self.strikes.index(self.strike_percent)]
          putPrice = [math.exp(-math.log(r/100+1)*t/252) * (k*self.hedge_percent*nminusd2 - f*self.hedge_percent*nminusd1) if not (D1tr[ind]==10000 or pd.isna(K[ind])) else (0 if (k - f) < 0 else (k - f)*self.hedge_percent) for ind, r, t, f, k, nminusd1, nminusd2, in zip(list(range(0,len(Rtr))), Rtr, Ttr, Ftr, Ktr, nMinusD1tr, nMinusD2tr)]
          putPriceFull = nANList + putPrice
          putPriceNAFill = list(pd.Series(putPriceFull).fillna(value=0))
          if self.position_type=='long':
               putPriceWithSpreads = [putPriceNAFill[ind]*(1+spreadMultipl) if daysToRenew[ind]==1 else (putPriceNAFill[ind]*(1-spreadMultipl) if daysToSell[ind]==1 and D1[ind]!=10000 else putPriceNAFill[ind]) for ind, x in enumerate(putPriceNAFill)]
          else:
               putPriceWithSpreads = [-putPriceNAFill[ind]*(1-spreadMultipl) if daysToRenew[ind]==1 else (-putPriceNAFill[ind]*(1+spreadMultipl) if daysToSell[ind]==1 and D1[ind]!=10000 else -putPriceNAFill[ind]) for ind, x in enumerate(putPriceNAFill)]
          return putPriceWithSpreads
     
     def is_in_money(self, daysToRenew, F, daysForOptionCalculation):
          '''Shows periods when the option is in money
          '''
          K = self.show_option_strikes(daysForOptionCalculation, daysToRenew, F)
          moneyness = [1 if K[ind]>=F[ind] else (float('nan') if pd.isna(K[ind]) else 0) for ind, x, y in zip(list(range(0,len(F))), F, K)]
          return moneyness







     