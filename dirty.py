bigZip = list(zip(futuresPrice, daysLeft[0], daysToRenew, strike, d1, d2, callMoneyness, putMoneyness, callPrice, putPrice))
bigDF = pd.DataFrame(bigZip, columns = ['Futur', 'DaysLeft', 'Renew_dates', 'Strike', 'd1', 'd2', 'Callmoney', 'Putmoney', 'callprice', 'putprice'])
bigDF.index = closePrice.index
bigDF.to_csv('./bigDF.csv', sep=';', decimal=',')

#the way comissions are included:
#they are country dependent 
comissions = [[0.1,0.05,0.02,0.01,0.01,0.01,0.02,0.05,0.1],[0.15,0.05,0.02,0.01,0.01,0.03,0.1,0.2,0.25]]

#time for code execution
import time
start_time = time.time()#goes to the upper side

print("--- %s seconds ---" % (time.time() - start_time))#goes down the code
  

#help with capital movement
a = [100,150,200,250,350,500,600,650,700,800,700,800,600,550,600,650,650]
b = [0,0,0,0,0,0,60,55,50,20,0,80,95,100,60,0,0]
c = [0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,0]
d = [0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0]
e = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0]

capitalBH = []
firstNotNull = pd.Series(a).notna().idxmax()
for i in range(0,len(c)):
     if pd.isna(a[i]):
          capitalBH.append(float('nan'))
     elif i==firstNotNull: #work on this part
          capitalBH.append(a[i])
     elif c[i]==0:
          if e[i]==1:
               capitalBH.append(capitalBH[i-1]+a[i]-a[i-1]+b[i])
          else:
               capitalBH.append(capitalBH[i-1]+a[i]-a[i-1])
     else:
          if d[i]==1:
               capitalBH.append(capitalBH[i-1]+a[i]-a[i-1]-b[i])
          else:
               capitalBH.append(capitalBH[i-1]+a[i]-a[i-1])


f = [x+y if e[ind]!=1 else x for ind, x, y in zip(list(range(0,len(e))), capitalBH, b)]

#write data sets with prices and fundamentals 
largestTrainData = max(trainSetNumOfObserv)
spotPricetest = np.empty((largestTrainData,closePrice.shape[1]))
for i in range(0, closePrice.shape[1]):
     spotPricetest[:,i] = list(closePrice.iloc[:trainSetNumOfObserv[i], i]) + [float('nan')] * (largestTrainData-trainSetNumOfObserv[i])
spotPricetest = pd.DataFrame(spotPricetest)
spotPricetest.index = list(closePrice.index)[:largestTrainData]
spotPricetest.to_csv('./Spot_price_test.csv', sep=';', decimal=',')

PBtest = np.empty((largestTrainData,indPB.shape[1]))
for i in range(0, indPB.shape[1]):
     PBtest[:,i] = list(indPB.iloc[:trainSetNumOfObserv[i], i]) + [float('nan')] * (largestTrainData-trainSetNumOfObserv[i])
PBtest = pd.DataFrame(PBtest)
PBtest.index = list(indPB.index)[:largestTrainData]
PBtest.to_csv('./PB_test.csv', sep=';', decimal=',')

PEtest = np.empty((largestTrainData,indPE.shape[1]))
for i in range(0, indPE.shape[1]):
     PEtest[:,i] = list(indPE.iloc[:trainSetNumOfObserv[i], i]) + [float('nan')] * (largestTrainData-trainSetNumOfObserv[i])
PEtest = pd.DataFrame(PEtest)
PEtest.index = list(indPE.index)[:largestTrainData]
PEtest.to_csv('./PE_test.csv', sep=';', decimal=',')

PStest = np.empty((largestTrainData,indPS.shape[1]))
for i in range(0, indPS.shape[1]):
     PStest[:,i] = list(indPS.iloc[:trainSetNumOfObserv[i], i]) + [float('nan')] * (largestTrainData-trainSetNumOfObserv[i])
PStest = pd.DataFrame(PStest)
PStest.index = list(indPS.index)[:largestTrainData]
PStest.to_csv('./PS_test.csv', sep=';', decimal=',')

#making data for visual analysis (spot price and fundamentals)
largestTrainData = max(trainSetNumOfObserv)
developedMarketstest = np.empty((largestTrainData, int(closePrice.shape[1]/2*5)))
for i in range(0, int(indPS.shape[1]/2)):
     developedMarketstest[:,i*5] = list(closePrice.iloc[:trainSetNumOfObserv[i], i]) + [float('nan')] * (largestTrainData-trainSetNumOfObserv[i])
     developedMarketstest[:,i*5 + 1] = list(indPB.iloc[:trainSetNumOfObserv[i], i]) + [float('nan')] * (largestTrainData-trainSetNumOfObserv[i])
     developedMarketstest[:,i*5 + 2] = list(indPE.iloc[:trainSetNumOfObserv[i], i]) + [float('nan')] * (largestTrainData-trainSetNumOfObserv[i])
     developedMarketstest[:,i*5 + 3] = list(indPS.iloc[:trainSetNumOfObserv[i], i]) + [float('nan')] * (largestTrainData-trainSetNumOfObserv[i])
     developedMarketstest[:,i*5 + 4] = [float('nan')] * (largestTrainData)
developedMarketstest = pd.DataFrame(developedMarketstest)
developedMarketstest.index = list(indPS.index)[:largestTrainData]
developedMarketstest.to_csv('./Developed_markets_test.csv', sep=';', decimal=',')

#example of option strategies calculation
putATM5 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRatesList, daysLeft[0], futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])

#%%
from multiprocessing import Process
import sys

rocket = 0

def func1():
    global rocket
    print ('start func1')
    while rocket < 100000000:
        rocket += 1
    print ('end func1')

def func2():
    global rocket
    print ('start func2')
    while rocket < 100000000:
        rocket += 1
    print ('end func2')

if __name__=='__main__':
    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()

x = func1()

#%%
from concurrent.futures import ProcessPoolExecutor


a = 0

def func1(a):
     b = a
     while b < 100000000:
          b += 1
     return b
    
def func2(a):
     b = a
     while b < 100000000:
         b += 1
     return b

pool = ProcessPoolExecutor(2)
future = pool.submit(func1, a)
print(future.done())
future.result()

x = func1(a)

#%%
from multiprocessing import Process

a = 0

def func1():
     print('func 1 running')
     b = a
     while b < 100000000:
          b += 1
     print('func 1 ended')
     return b

def func2():
     print('func 2 running')
     b = a
     while b < 100000000:
         b += 1
     print('func 2 ended')
     return b

if __name__=='__main__':
     p1 = Process(target = func1)
     p1.start()
     p2 = Process(target = func2)
     p2.start()
     
#%%
'''
from multiprocessing import Process
#import sys

if __name__=='__main__':
    p1 = Process(target = type1_sharpes)
    p1.start()
    p2 = Process(target = type2_sharpes)
    p2.start()
    
#%%
from multiprocessing import Pool
pool = Pool()
result1 = pool.apply_async(type1_sharpes, [pBook])    # evaluate "solve1(A)" asynchronously
result2 = pool.apply_async(type1_sharpes, [pBook])    # evaluate "solve2(B)" asynchronously
answer1 = result1.get(timeout=10)
answer2 = result2.get(timeout=10)
'''
#%%
from concurrent.futures import ThreadPoolExecutor, Future

cores = ThreadPoolExecutor(max_workers=3)

pBT1SH = cores.submit(type1_sharpes, pBook)
pBT2SH = cores.submit(type2_sharpes, pBook)

#%%
from multiprocessing import Process
def main():
    p1 = Process(target = type1_sharpes, args=(pBook,))
    p2 = Process(target = type2_sharpes, args =(pBook,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

main()

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
          spreadMultipl = self.spreads[0 if self.countrynum<=9 else 1][self.strikes.index(self.strike_percent)]
          callPrice = [math.exp(-math.log(r/100+1)*t/252) * (f*self.hedge_percent*nd1 - k*self.hedge_percent*nd2) if not (D1[ind]==10000 or pd.isna(K[ind])) else (0 if (f - k) < 0 else (f - k)*self.hedge_percent)  for ind, r, t, f, k, nd1, nd2, in zip(list(range(0,len(R))), R, T, F, K, nD1, nD2)]
          callPriceNAFill = list(pd.Series(callPrice).fillna(value=0))
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

#%% OLD FORMAT OF COMBINATIONS CALCULATION

#This will be the part of a big loop (each for one type of a particular FA indicator) - P/B with type 1 condition
#try to create a function that will be called later in the loop
'''
def type1_sharpes(indFA):
     print('Processing TYPE 1 condition')
     seqFA = list(np.linspace(np.nanmin(pBook),np.nanmax(pBook),6)[:-1])
     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(30, 60, 4))
     seqAD = list(np.linspace(5, 30, 6))
     seqKAMA = list(np.linspace(30, 60, 4))
     seqMACD = list(np.linspace(30, 60, 4))
     seqTRIX = list(np.linspace(15, 30, 4))
     comb = list(itertools.product(seqFA,seqEMAShort, seqEMALong, seqAD, seqKAMA, seqMACD, seqTRIX))
     sharpeMatrix =  np.empty((len(comb), 16))
     
     for i in range(0, nIter): #len(comb))
          if i % 1 == 0:
               print('Calculating {}`th iteration'.format(i+1))
          else:
               pass

          condFARaw = [1 if x>comb[i][0] else (float('nan') if pd.isna(x) else 0) for x in indFA]#condition to hedge determined by FA indicator
          condFAType1 = get_condition_type1(condFARaw, daysTillExpirationTr)

          eMACond = EMA_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][1]), int(comb[i][2])) #works
          fAT1PlusEMA = [x*y for x, y in zip(condFAType1,eMACond)]
          datesTuple = dates_tuple(fAT1PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
          optionStrategies = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT1Portfolio1 = first_portfolios_with_given_options(spotPrice, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

          aDCond = AD_sign(highPriceDF.iloc[:trainLength,countrynum], lowPriceDF.iloc[:trainLength,countrynum], 5, 34, int(comb[i][3])) #works
          fAT1PlusAD = [x*y for x, y in zip(condFAType1,aDCond)]
          datesTuple = dates_tuple(fAT1PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies2 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT1Portfolio2 = portfolios_with_given_options(fAT1Portfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)

          kAMACond = KAMA_sign(closePriceDF.iloc[:trainLength,countrynum], 10, 2, 30, int(comb[i][4])) #works
          fAT1PlusKAMA = [x*y for x, y in zip(condFAType1,kAMACond)]
          datesTuple = dates_tuple(fAT1PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies3 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT1Portfolio3 = portfolios_with_given_options(fAT1Portfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

          mACDCond = MACD_sign(closePriceDF.iloc[:trainLength,countrynum], 26, 12, int(comb[i][5])) #works
          fAT1PlusMACD = [x*y for x, y in zip(condFAType1,mACDCond)]
          datesTuple = dates_tuple(fAT1PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies4 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT1Portfolio4 = portfolios_with_given_options(fAT1Portfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

          tRIXCond = TRIX_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][6])) #works
          fAT1PlusTRIX = [x*y for x, y in zip(condFAType1,tRIXCond)]
          datesTuple = dates_tuple(fAT1PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies5 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT1Portfolio5 = portfolios_with_given_options(fAT1Portfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

          fAT1SharpeRatiosVector = sharpe_ratios_vector(fAT1Portfolio5)#vector taht will be futher added to a big matrix
          sharpeMatrix[i,:] = fAT1SharpeRatiosVector.transpose()
     print('TYPE 1 condition processed')
     return sharpeMatrix
     
#%%P/B with type 2 signal
def type2_sharpes(indFA):
     print('Processing TYPE 2 condition')
     seqFA = list(np.linspace(np.nanmin(pBook),np.nanmax(pBook),6)[:-1])
     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(30, 60, 4))
     seqAD = list(np.linspace(5, 30, 6))
     seqKAMA = list(np.linspace(30, 60, 4))
     seqMACD = list(np.linspace(30, 60, 4))
     seqTRIX = list(np.linspace(15, 30, 4))
     comb = list(itertools.product(seqFA,seqEMAShort, seqEMALong, seqAD, seqKAMA, seqMACD, seqTRIX))
     sharpeMatrix =  np.empty((len(comb), 16))
     
     for i in range(0, nIter): #len(comb))
          if i % 1 == 0:
               print('Calculating {}`th iteration'.format(i+1))
          else:
               pass
          
          condFARaw = [1 if x>comb[i][0] else (float('nan') if pd.isna(x) else 0) for x in indFA]#condition to hedge determined by FA indicator
          condFAType2 = get_condition_type2(condFARaw)

          eMACond = EMA_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][1]), int(comb[i][2])) #works
          fAT2PlusEMA = [x*y for x, y in zip(condFAType2,eMACond)]
          datesTuple = dates_tuple(fAT2PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
          optionStrategies = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT2Portfolio1 = first_portfolios_with_given_options(spotPrice, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

          aDCond = AD_sign(highPriceDF.iloc[:trainLength,countrynum], lowPriceDF.iloc[:trainLength,countrynum], 5, 34, int(comb[i][3])) #works
          fAT2PlusAD = [x*y for x, y in zip(condFAType2,aDCond)]
          datesTuple = dates_tuple(fAT2PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies2 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT2Portfolio2 = portfolios_with_given_options(fAT2Portfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)

          kAMACond = KAMA_sign(closePriceDF.iloc[:trainLength,countrynum], 10, 2, 30, int(comb[i][4])) #works
          fAT2PlusKAMA = [x*y for x, y in zip(condFAType2,kAMACond)]
          datesTuple = dates_tuple(fAT2PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies3 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT2Portfolio3 = portfolios_with_given_options(fAT2Portfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

          mACDCond = MACD_sign(closePriceDF.iloc[:trainLength,countrynum], 26, 12, int(comb[i][5])) #works
          fAT2PlusMACD = [x*y for x, y in zip(condFAType2,mACDCond)]
          datesTuple = dates_tuple(fAT2PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies4 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT2Portfolio4 = portfolios_with_given_options(fAT2Portfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

          tRIXCond = TRIX_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][6])) #works
          fAT2PlusTRIX = [x*y for x, y in zip(condFAType2,tRIXCond)]
          datesTuple = dates_tuple(fAT2PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies5 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT2Portfolio5 = portfolios_with_given_options(fAT2Portfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

          fAT2SharpeRatiosVector = sharpe_ratios_vector(fAT2Portfolio5)#vector that will be futher added to a big matrix
          sharpeMatrix[i,:] = fAT2SharpeRatiosVector.transpose()
     print('TYPE 2 condition processed')
     return sharpeMatrix

#%%P/B with type 3 signal
def type3_sharpes(indFA):
     print('Processing TYPE 3 condition')
     seqFA = list(np.linspace(np.nanmin(pBook),np.nanmax(pBook),6)[:-1])
     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(30, 60, 4))
     seqAD = list(np.linspace(5, 30, 6))
     seqKAMA = list(np.linspace(30, 60, 4))
     seqMACD = list(np.linspace(30, 60, 4))
     seqTRIX = list(np.linspace(15, 30, 4))
     comb = list(itertools.product(seqFA,seqEMAShort, seqEMALong, seqAD, seqKAMA, seqMACD, seqTRIX))
     sharpeMatrix =  np.empty((len(comb), 16))
     
     for i in range(0, nIter): #len(comb))
          if i % 1 == 0:
               print('Calculating {}`th iteration'.format(i+1))
          else:
               pass
     
          condFARaw = [1 if x>comb[i][0] else (float('nan') if pd.isna(x) else 0) for x in indFA]#condition to hedge determined by FA indicator
          condFAType3 = get_condition_type3(indFA, condFARaw)

          eMACond = EMA_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][1]), int(comb[i][2])) #works
          fAT3PlusEMA = [x*y for x, y in zip(condFAType3,eMACond)]
          datesTuple = dates_tuple(fAT3PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
          optionStrategies = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio1 = first_portfolios_with_given_options(spotPrice, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

          aDCond = AD_sign(highPriceDF.iloc[:trainLength,countrynum], lowPriceDF.iloc[:trainLength,countrynum], 5, 34, int(comb[i][3])) #works
          fAT3PlusAD = [x*y for x, y in zip(condFAType3,aDCond)]
          datesTuple = dates_tuple(fAT3PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies2 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio2 = portfolios_with_given_options(fAT3Portfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)

          kAMACond = KAMA_sign(closePriceDF.iloc[:trainLength,countrynum], 10, 2, 30, int(comb[i][4])) #works
          fAT3PlusKAMA = [x*y for x, y in zip(condFAType3,kAMACond)]
          datesTuple = dates_tuple(fAT3PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies3 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio3 = portfolios_with_given_options(fAT3Portfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

          mACDCond = MACD_sign(closePriceDF.iloc[:trainLength,countrynum], 26, 12, int(comb[i][5])) #works
          fAT3PlusMACD = [x*y for x, y in zip(condFAType3,mACDCond)]
          datesTuple = dates_tuple(fAT3PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies4 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio4 = portfolios_with_given_options(fAT3Portfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

          tRIXCond = TRIX_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][6])) #works
          fAT3PlusTRIX = [x*y for x, y in zip(condFAType3,tRIXCond)]
          datesTuple = dates_tuple(fAT3PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies5 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT3Portfolio5 = portfolios_with_given_options(fAT3Portfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators
     
          fAT3SharpeRatiosVector = sharpe_ratios_vector(fAT3Portfolio5)
          sharpeMatrix[i,:] = fAT3SharpeRatiosVector.transpose()
     print('TYPE 3 condition processed')
     return sharpeMatrix

#%%P/B with type 4 signal
def type4_sharpes(indFA):
     print('Processing TYPE 4 condition')
     seqFA = list(np.linspace(np.nanmin(pBook),np.nanmax(pBook),6)[:-1])
     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(30, 60, 4))
     seqAD = list(np.linspace(5, 30, 6))
     seqKAMA = list(np.linspace(30, 60, 4))
     seqMACD = list(np.linspace(30, 60, 4))
     seqTRIX = list(np.linspace(15, 30, 4))
     comb = list(itertools.product(seqFA,seqEMAShort, seqEMALong, seqAD, seqKAMA, seqMACD, seqTRIX))
     sharpeMatrix =  np.empty((len(comb), 16))
     
     for i in range(0, nIter): #len(comb))
          if i % 1 == 0:
               print('Calculating {}`th iteration'.format(i+1))
          else:
               pass
          
          condFARaw = [1 if x>comb[i][0] else (float('nan') if pd.isna(x) else 0) for x in indFA]#condition to hedge determined by FA indicator

          eMACond = EMA_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][1]), int(comb[i][2])) #works
          condFAType4EMA = get_condition_type4(condFARaw, eMACond)
          fAT4PlusEMA = [x*y for x, y in zip(condFAType4EMA,eMACond)]
          datesTuple = dates_tuple(fAT4PlusEMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
          optionStrategies = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT4Portfolio1 = first_portfolios_with_given_options(spotPrice, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

          aDCond = AD_sign(highPriceDF.iloc[:trainLength,countrynum], lowPriceDF.iloc[:trainLength,countrynum], 5, 34, int(comb[i][3])) #works
          condFAType4AD = get_condition_type4(condFARaw, aDCond)
          fAT4PlusAD = [x*y for x, y in zip(condFAType4AD,aDCond)]
          datesTuple = dates_tuple(fAT4PlusAD, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies2 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT4Portfolio2 = portfolios_with_given_options(fAT4Portfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)

          kAMACond = KAMA_sign(closePriceDF.iloc[:trainLength,countrynum], 10, 2, 30, int(comb[i][4])) #works
          condFAType4KAMA = get_condition_type4(condFARaw, kAMACond)
          fAT4PlusKAMA = [x*y for x, y in zip(condFAType4KAMA,kAMACond)]
          datesTuple = dates_tuple(fAT4PlusKAMA, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies3 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT4Portfolio3 = portfolios_with_given_options(fAT4Portfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

          mACDCond = MACD_sign(closePriceDF.iloc[:trainLength,countrynum], 26, 12, int(comb[i][5])) #works
          condFAType4MACD = get_condition_type4(condFARaw, mACDCond)
          fAT4PlusMACD = [x*y for x, y in zip(condFAType4MACD,mACDCond)]
          datesTuple = dates_tuple(fAT4PlusMACD, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies4 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT4Portfolio4 = portfolios_with_given_options(fAT4Portfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

          tRIXCond = TRIX_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][6])) #works
          condFAType4TRIX = get_condition_type4(condFARaw, tRIXCond)
          fAT4PlusTRIX = [x*y for x, y in zip(condFAType4TRIX,tRIXCond)]
          datesTuple = dates_tuple(fAT4PlusTRIX, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies5 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT4Portfolio5 = portfolios_with_given_options(fAT4Portfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

          fAT4SharpeRatiosVector = sharpe_ratios_vector(fAT4Portfolio5)
          sharpeMatrix[i,:] = fAT4SharpeRatiosVector.transpose()
     print('TYPE 4 condition processed')
     return sharpeMatrix

#%%P/B with type 2 signal
def type5_sharpes():
     print('Processing TYPE 5 condition')
     seqEMAShort = list(np.linspace(5, 20, 4))
     seqEMALong = list(np.linspace(30, 60, 4))
     seqAD = list(np.linspace(5, 30, 6))
     seqKAMA = list(np.linspace(30, 60, 4))
     seqMACD = list(np.linspace(30, 60, 4))
     seqTRIX = list(np.linspace(15, 30, 4))
     comb = list(itertools.product(seqEMAShort, seqEMALong, seqAD, seqKAMA, seqMACD, seqTRIX))
     sharpeMatrix =  np.empty((len(comb), 16))
     
     for i in range(0, nIter): #len(comb))
          if i % 1 == 0:
               print('Calculating {}`th iteration'.format(i+1))
          else:
               pass

          eMACond = EMA_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][0]), int(comb[i][1])) #works
          datesTuple = dates_tuple(eMACond, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
          optionStrategies = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT5Portfolio1 = first_portfolios_with_given_options(spotPrice, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)

          aDCond = AD_sign(highPriceDF.iloc[:trainLength,countrynum], lowPriceDF.iloc[:trainLength,countrynum], 5, 34, int(comb[i][2])) #works
          datesTuple = dates_tuple(aDCond, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies2 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT5Portfolio2 = portfolios_with_given_options(fAT5Portfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)

          kAMACond = KAMA_sign(closePriceDF.iloc[:trainLength,countrynum], 10, 2, 30, int(comb[i][3])) #works
          datesTuple = dates_tuple(kAMACond, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies3 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT5Portfolio3 = portfolios_with_given_options(fAT5Portfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)

          mACDCond = MACD_sign(closePriceDF.iloc[:trainLength,countrynum], 26, 12, int(comb[i][4])) #works
          datesTuple = dates_tuple(mACDCond, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies4 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT5Portfolio4 = portfolios_with_given_options(fAT5Portfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)

          tRIXCond = TRIX_sign(closePriceDF.iloc[:trainLength,countrynum], int(comb[i][5])) #works
          datesTuple = dates_tuple(tRIXCond, daysTillExpirationTr, daysOfExpirationTr, daysToRenew)
          optionStrategies5 = calculate_strategies(countrynum, interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])   
          fAT5Portfolio5 = portfolios_with_given_options(fAT5Portfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators

          fAT5SharpeRatiosVector = sharpe_ratios_vector(fAT5Portfolio5)#vector that will be futher added to a big matrix
          sharpeMatrix[i,:] = fAT5SharpeRatiosVector.transpose()
     print('TYPE 5 condition processed')
     return sharpeMatrix
'''

#%%obtaining matrices for different FA indicators and types of signals
'''
nIter = 1
start_time = time.time()
pBT1Sharpes = type1_sharpes(pBook)

pET1Sharpes = type1_sharpes(pEarn)
pST1Sharpes = type1_sharpes(pSale)

pBT2Sharpes = type2_sharpes(pBook)
pET2Sharpes = type2_sharpes(pEarn)
pST2Sharpes = type2_sharpes(pSale)

pBT3Sharpes = type3_sharpes(pBook)
pET3Sharpes = type3_sharpes(pEarn)
pST3Sharpes = type3_sharpes(pSale)

pBT4Sharpes = type4_sharpes(pBook)
pET4Sharpes = type4_sharpes(pEarn)
pST4Sharpes = type4_sharpes(pSale)

t5Sharpes = type5_sharpes()

print("--- %s seconds ---" % (time.time() - start_time))#goes down the code
#'''

#%%calculation decomposition
'''
checkCond = [1]*len(interpolatedRatesList)
datesTuple = dates_tuple(checkCond, daysTillExpirationTr, daysOfExpirationTr, daysToRenewInit)

start_time = time.time()
putATM5 = PutOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
callATM5 = CallOpt(100, 0.2, 'long', countrynum).option_price(interpolatedRatesList, daysTillExpirationTr, futuresPrice, vixList, datesTuple[0], datesTuple[1], datesTuple[3])
print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(putATM5)
'''
#write logics and description for files best_portfolios, condition_types

#%%plots with portfolios on test set 
'''
countrynum=2
numOfObserv = testSetNumOfObserv[countrynum]
closePriceSeries = closePriceDF.iloc[:,countrynum]
lowPriceSeries = lowPriceDF.iloc[:,countrynum]
highPriceSeries = highPriceDF.iloc[:,countrynum]
listOfParams = [5.0, 50.0, 5.0, 60.0, 30.0, 25.0]
numOfBestStrategy = 0
closePriceList = list(closePriceSeries)
interpolatedRatesList = list(interpolatedRates.iloc[:,countrynum]) 
daysTillExpiration = daysLeft[0]
vixList = list(vix.iloc[:,countrynum])
daysOfExpiration = daysLeft[1] 
closeList = list(closePriceSeries[-numOfObserv:])
     
futuresPrice = futures_price(interpolatedRatesList[-numOfObserv:], daysTillExpiration[-numOfObserv:], list(closePriceSeries)[-numOfObserv:])
daysToRenewInit = new_futures_dates(daysOfExpiration[-numOfObserv:], futuresPrice)

#eMACond = EMA_sign(closePriceSeries, int(listOfParams[0]), int(listOfParams[1]))[-numOfObserv:]
eMACond = EMA_sign(closePriceSeries, 30, 150)[-numOfObserv:]
#laggedClose = list(closePriceSeries.shift(50))
#monthDeltaClosePrice = [(x-y)/y for x, y in zip(closePriceList, laggedClose)]
#fallCondition = [1 if x<=0 else (0 if x>0 else float('nan')) for x in monthDeltaClosePrice]
#eMAplusfall = [x*y for x, y in zip(fallCondition[-numOfObserv:], eMACond)]
datesTuple = dates_tuple(eMACond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])#daysToRenewFinal, daysToSellFInal, daysToHoldOptions, daysForOptionCalculation respectively
optionStrategies = calculate_strategies(countrynum,interpolatedRatesList[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
fAPortfolio1 = first_portfolios_with_given_options(closePriceList[-numOfObserv:], datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies)
firstPortf = list(fAPortfolio1.iloc[:,0])
put90 = list(fAPortfolio1.iloc[:,2])
#x = firstPortf

plt.plot(firstPortf)
plt.plot(list(closePriceSeries)[-numOfObserv:])
plt.plot(put90)
plt.plot([x*3000 for x in eMACond])

fAPortfolio1.plot()
plt.legend()

aDCond = AD_sign(highPriceSeries, lowPriceSeries, 5, 34, int(listOfParams[2]))[-numOfObserv:]
datesTuple = dates_tuple(aDCond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
optionStrategies2 = calculate_strategies(countrynum,interpolatedRatesList[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
fAPortfolio2 = portfolios_with_given_options(fAPortfolio1, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies2)
secondPortf = list(fAPortfolio2.iloc[:,0])

plt.plot(secondPortf)
plt.plot(list(closePriceSeries)[-numOfObserv:])

kAMACond = KAMA_sign(closePriceSeries, 10, 2, 30, int(listOfParams[3]))[-numOfObserv:]
datesTuple = dates_tuple(kAMACond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
optionStrategies3 = calculate_strategies(countrynum,interpolatedRatesList[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
fAPortfolio3 = portfolios_with_given_options(fAPortfolio2, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies3)
thirdPortf = list(fAPortfolio3.iloc[:,0])

plt.plot(thirdPortf)
plt.plot(list(closePriceSeries)[-numOfObserv:])

mACDCond = MACD_sign(closePriceSeries, 26, 12, int(listOfParams[4]))[-numOfObserv:]
datesTuple = dates_tuple(mACDCond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
optionStrategies4 = calculate_strategies(countrynum,interpolatedRatesList[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
fAPortfolio4 = portfolios_with_given_options(fAPortfolio3, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies4)
forthPortf = list(fAPortfolio4.iloc[:,0])

plt.plot(forthPortf)
plt.plot(list(closePriceSeries)[-numOfObserv:])

tRIXCond = TRIX_sign(closePriceSeries, int(listOfParams[5]))[-numOfObserv:]
datesTuple = dates_tuple(tRIXCond, daysTillExpiration[-numOfObserv:], daysOfExpiration[-numOfObserv:], daysToRenewInit[-numOfObserv:])
optionStrategies5 = calculate_strategies(countrynum,interpolatedRatesList[-numOfObserv:], daysTillExpiration[-numOfObserv:], futuresPrice, vixList[-numOfObserv:], datesTuple[0], datesTuple[1], datesTuple[3])   
fAPortfolio5 = portfolios_with_given_options(fAPortfolio4, datesTuple[2], datesTuple[0], datesTuple[1], optionStrategies5)#this is the final set of portfolios for given FA condition type, given FA indicator value, set of TA indicators
fifthPortf = list(fAPortfolio5.iloc[:,0])

plt.plot(fifthPortf)
plt.plot(list(closePriceSeries)[-numOfObserv:])

plt.plot(list(closePriceSeries)[-numOfObserv:])
plt.plot(firstPortf)
plt.plot(secondPortf)
plt.plot(thirdPortf)
plt.plot(forthPortf)
plt.plot(fifthPortf)

fASharpeRatiosVector = sharpe_ratios_vector(fAPortfolio5)
bestTrainStrategyCapital = list(fAPortfolio5.iloc[:,numOfBestStrategy])
bestTrainStrategySharpe = float(fASharpeRatiosVector.iloc[numOfBestStrategy,0])
bestTestSharpe = float(fASharpeRatiosVector.max())
'''

#%%

#AD analysis (add new periods)
awesomeOsc = ta.momentum.AwesomeOscillatorIndicator(high=highPriceSeries, low=lowPriceSeries, s=5, len=34, fillna=False)._ao
awOscSMA = awesomeOsc.rolling(window=70).mean()

aOList = list(awesomeOsc)[-numOfObserv:]
aOSMAList = list(awOscSMA)[-numOfObserv:]

accelDeccel = [1 if (x - y) < 0 else 0 for x, y in zip(awesomeOsc,awOscSMA)][-numOfObserv:]

#plt.plot(aOList)
plt.plot([x*4500 for x in accelDeccel])
plt.plot(closeList)

#%%
#TRIX analysis (add new periods)
tRIX = list(ta.trend.TRIXIndicator(close=closePriceSeries, n=15, fillna=False)._trix)[-numOfObserv:]
tRIXSign = [1 if x < 0 else 0 for x in tRIX]

plt.plot([x*4500 for x in tRIXSign])
plt.plot(closeList)
     
#%%
#KAMA analysis
aMA = ta.momentum.KAMAIndicator(close=closePriceSeries, n=10, pow1=2, pow2=30, fillna=False)._kama
aMASMA = pd.Series(aMA).rolling(window=80).mean()
aMAList = list(aMA)[-numOfObserv:]
aMASMAList = list(aMASMA)[-numOfObserv:]
aMASign = [1 if (x - y) < 0 else 0 for x, y in zip(aMAList, aMASMAList)]

plt.plot(closeList)
plt.plot(aMAList)
plt.plot(aMASMAList)
plt.plot([x*5000 for x in aMASign])

#%%how each best TA works on France
fig,ax = plt.subplots()
ax.plot(list(closePriceSeries)[-1200:], label = '')
ax2=ax.twinx()
ax2.plot(EMA_sign(closePriceSeries, 5, 80)[-1200:],color="red", label = '')
ax2.set_ylim(0,6)

fig,ax = plt.subplots()
ax.plot(list(closePriceSeries)[-1200:-700], label = '')
ax2=ax.twinx()
ax2.plot(AD_sign(highPriceSeries, lowPriceSeries, 5, 34, 5)[-1200:-700],color="red", label = '')
ax2.set_ylim(0,6)

fig,ax = plt.subplots()
ax.plot(list(closePriceSeries)[-1200:], label = '')
ax2=ax.twinx()
ax2.plot(KAMA_sign(closePriceSeries, 10, 2, 30, 100)[-1200:],color="red", label = '')
ax2.set_ylim(0,6)

fig,ax = plt.subplots()
ax.plot(list(closePriceSeries)[-1200:], label = '')
ax2=ax.twinx()
ax2.plot(MACD_sign(closePriceSeries, 26, 12, 30)[-1200:],color="red", label = '')
ax2.set_ylim(0,6)

fig,ax = plt.subplots()
ax.plot(list(closePriceSeries)[-1200:], label = '')
ax2=ax.twinx()
tRIXPlot = TRIX_sign(closePriceSeries, 60)[-1200:]
ax.plot([x*y for x, y in zip(list(closePriceSeries)[-1200:], tRIXPlot)],color="red", label = '')
ax.set_ylim(3000,6000)
#ax2.set_ylim(0,6)

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load a numpy structured array from yahoo csv data with fields date, open,
# close, volume, adj_close from the mpl-data/example directory.  This array
# stores the date as an np.datetime64 with a day unit ('D') in the 'date'
# column.
with cbook.get_sample_data('goog.npz') as datafile:
    data = np.load(datafile)['price_data']

fig, ax = plt.subplots()
ax.plot('date', 'adj_close', data=data)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(data['date'][0], 'Y')
datemax = np.datetime64(data['date'][-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

plt.show()