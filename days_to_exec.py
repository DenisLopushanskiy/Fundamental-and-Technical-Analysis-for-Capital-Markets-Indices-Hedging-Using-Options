import pandas as pd
import copy
#%% function that creates list of days left to execution
def days_left(givenDF):
     '''Calculates list days till execution for futures and options [0] and gives list with units assigned to days of execution.
     :givenDF: any of initial DF in analysis
     :return: tuple with lists.
     '''
     month = pd.DatetimeIndex(givenDF.index).month
     exDateList = [None]*len(month)
     for i in range(0, len(month)-1):
          if month[i]==3 and month[i+1]==4:
               exDateList[i] = 1
          elif month[i]==6 and month[i+1]==7:
               exDateList[i] = 1
          elif month[i]==9 and month[i+1]==10:
               exDateList[i] = 1
          elif month[i]==12 and month[i+1]==1:
               exDateList[i] = 1
          else:
               exDateList[i] = 0
     exDateList[-1] = 1 #it is known that the last day is 31/12/2018, so it is exercise day
     given_dates = [i for i in range(len(exDateList)) if exDateList[i] == 1]
     daysNum = list(range(0,len(exDateList)))
     daysLeft = [None]*len(exDateList)
     for i in range(0, len(exDateList)):
          daysLeftList = [x-daysNum[i] for x in given_dates]
          daysLeft[i] = min(list(filter(lambda x: x >= 0, daysLeftList)))
     return daysLeft, exDateList

#%% function that shows dates, when derivatives are renewed  
def new_futures_dates(givenList, F):
     '''Calculates days when new futures and options appear. 
     :givenList: any DF from analysis
     :F: list with futures price
     :return: list with units when new futures/options appear
     '''
     renewPositionDays = [x for x in (list(pd.Series(givenList).shift(1)))]
     renewPositionDays[0] = 1 #it is just known that the first date is 01.01.1980 (this is data sensitive, some more advanced calendarr should be used)
     renewPositionDays = [int(x) for x in renewPositionDays]
     firstFuturesDay = next(x for x, val in enumerate(F) if pd.notna(val)) #the first day, when futures price exists, so option on it can be bought 
     renewPositionDays[firstFuturesDay] = 1 #assign 1 to that day
     return renewPositionDays

#%%
def give_first_one(x: list):
    '''This function searches the indexes of first 1's elements in the sequence of 1's and 0's in list of conditions to hedge (the first 1 after the 0)
    :sequence: the list of 1's and 0's
    :return: y - the list of indexes 
    '''
    if not isinstance(x, list):
        raise ValueError('The sequence must be the python\'s list object')
    y = []
    y.append(x[0])
    for j in range(1, len(x)):
        if x[j] == 1 and x[j] != x[j-1]:
             y.append(1)
        elif pd.isna(x[j]):
            y.append(float('nan'))
        else:
            y.append(0)
    return y

#%%
def give_first_zero(x: list):
    '''This function searches the indexes of first 0's elements in the sequence of 1's and 0's in list of conditions to hedge (the first 0 after the 1)
    :sequence: the list of 1's and 0's
    :return: ans - the list of indexes 
    '''
    if not isinstance(x, list):
        raise ValueError('The sequence must be the python\'s list object')
    y = []
    y.append(x[0])
    for j in range(1, len(x)):
        if x[j] == 0 and x[j] != x[j-1] and pd.notna(x[j-1]):
             y.append(1)
        elif x[j] == 0 and x[j] != x[j-1] and pd.isna(x[j-1]):
            y.append(0)
        elif pd.isna(x[j]):
            y.append(float('nan'))
        else:
            y.append(0)
    return y

#%%
def period_to_hold_options(daysToRenewFinal, daysToSellFinal, daysTillExecution):
     '''Function that gives interval with units when the position is bought and hold (without day to sell). Other values are NaN.
     :daysToRenewFinal: list that gives dates when to buy options (and to renew them) inclusing signals from FA and TA
     :daysToSellFinal: list that gives dates when to sell position (including day of expiration)
     :daysTillExecution: use daysLeft[0] - periods to execution
     :return: list of days when to buy and hold position (without sale day)
     '''
     derivativePosition = []
     derivativePosition.append(1 if daysToRenewFinal[0]==1 and daysToSellFinal[0]==0 else 0)
     for i in range(1, len(daysToRenewFinal)):
          if derivativePosition[i-1]==0:
               if daysToRenewFinal[i]==1 and daysToSellFinal[i]==0:
                    derivativePosition.append(1)
               else:
                    derivativePosition.append(0)
          else:
               if daysToRenewFinal[i]==0 and daysToSellFinal[i]==1:
                    derivativePosition.append(0)
               else:
                    derivativePosition.append(1)
     return derivativePosition

#%%
def dates_tuple(condFATA, daysTillExecution, executionDays, daysToRenew):
     '''Gives days to reconsider position and dates for option calculation (for which dates to calculate)
     :condFATA: 1 when FA and TA give condition to hedge
     :daysTillExecution: daysLeft[0]
     :executionDays: daysLeft[1]
     :daysToRenew: initial first days when to reconsider position
     :return: tuple: final list with days to reconsider and execute position, and period of days for option to hold and calculate
     '''
     daysToSellFATA = give_first_zero(condFATA)#days when FA+TA say to sell options
     daysToSellFiltered = [0 if (z==0 or pd.isna(z)) else x for x, z in zip(executionDays, condFATA)]#excluding expiration dates when whe system says not to hedge by FA+TA
     daysToSellFinal = [0 if pd.isna(y) else (x+y if x+y!=2 else 1) for x, y in zip(daysToSellFiltered, daysToSellFATA)]#days inclusing FA+TA system sell dates and (expiration dates for the interval when it is necessary to hedge by FA+TA)
     daysToRenewFiltered = [0 if (y==0 or pd.isna(y)) else x for x, y in zip(daysToRenew, condFATA)]#deletes units when FA+TA is 0 - no need to renew
     daysToRenewFATA = give_first_one(condFATA)#days when FA+TA say to buy options 
     daysToRenewFinal = [0 if (pd.isna(y) or daysTillExecution[ind]==0) else (x+y if x+y!=2 else 1) for ind, x, y in zip(list(range(0,len(daysTillExecution))), daysToRenewFiltered, daysToRenewFATA)]#days inclusing FA+TA system buy dates and (renew dates for the interval when it is necessary to hedge by FA+TA)
     daysToHoldOptions = period_to_hold_options(daysToRenewFinal, daysToSellFinal, daysTillExecution)#periods, for which option position is held 
     daysForOptionCalculation = [x+y if not x+y==2 else x for x, y in zip(daysToHoldOptions, daysToSellFinal)]#based on period_to_hold_options but also included days of expiration and days when the derivatives are sold for option price calculation intervals
     return daysToRenewFinal, daysToSellFinal, daysToHoldOptions, daysForOptionCalculation

#%%
def get_condition_type1(FACondRaw, daysTillExecution):
     '''Upgrades raw condition from FA indicator. Adds periods to the signal before the expiration date (type 1 condition).
     :FACondRaw: initital condition from FA indicator
     :daysTillExecution: daysLeft[0] usually
     :output: list with upgraded periods for option calculation
     '''
     firstZeros = [ind for ind, x in enumerate(FACondRaw) if (FACondRaw[ind]==0 and FACondRaw[ind-1]==1)]#where to add list with units
     periodForSubsitution = [daysTillExecution[x] for x in firstZeros]#how many units to add to initial condition
     condFAType1 = copy.copy(FACondRaw)
     for ind, x in enumerate(firstZeros):
          condFAType1[x:x+periodForSubsitution[ind]] = [1] * periodForSubsitution[ind]
          condFAType1 = condFAType1[0:len(FACondRaw)]#in the case if length of substitution goes further than the sample size
     return condFAType1

#%%
def get_condition_type2(FACondRaw):
     '''Upgrades raw condition from FA indicator. Adds 32 periods to the signal before the expiration (type 2 condition).
     :FACondRaw: initital condition from FA indicator
     :output: list with upgraded periods for option calculation
     '''
     firstZeros = [ind for ind, x in enumerate(FACondRaw) if (FACondRaw[ind]==0 and FACondRaw[ind-1]==1)]#where to add list with units
     periodForSubsitution = 32#32 days (1.5 month) to add to initial condition
     condFAType2 = copy.copy(FACondRaw)
     for ind, x in enumerate(firstZeros):
          condFAType2[x:x + periodForSubsitution] = [1] * periodForSubsitution
          condFAType2 = condFAType2[0:len(FACondRaw)]#in the case if length of substitution goes further than the sample size
     return condFAType2

#%%
def get_condition_type3(indFA, FACondRaw):
     '''Upgrades raw condition from FA indicator. Adds periods when lagged FA indicator is higher (delta is negative) - type 3 condition.
     :indFA: list with initial FA indicator
     :FACondRaw: initital condition from FA indicator
     :output: list with upgraded periods for option calculation
     '''
     #indFAYesterday = list(pd.Series(indFA).shift(1))#lag of 1 day is used to avoid using 'uture' information (when in the morning we know if at the close price PB crossed the threshold)
     laggedPBook = list(pd.Series(indFA).shift(21))
     monthDeltaPB = [(x-y)/y for x, y in zip(indFA, laggedPBook)]
     condPBType3 = []
     condPBType3.append(FACondRaw[0])
     for i in range(1, len(FACondRaw)):
          if FACondRaw[i]==1:
               condPBType3.append(1)
          elif FACondRaw[i]==0:
               if condPBType3[i-1]==1 and monthDeltaPB[i]<0:
                    condPBType3.append(1)
               else:
                    condPBType3.append(0)
          else:
               condPBType3.append(float('nan'))
     return condPBType3

#%%
def get_condition_type4(FACondRaw, condTA):
     '''Upgrades raw condition from FA indicator. Adds periods when TA indicator still gives signal to buy option strategy - type 4 condition.
     :indFA: list with initial FA indicator
     :indTA: list with signals from TA
     :output: list with upgraded periods for option calculation
     '''
     condPBType4 = []
     condPBType4.append(FACondRaw[0])
     for i in range(1, len(FACondRaw)):
          if FACondRaw[i]==1:
               condPBType4.append(1)
          elif FACondRaw[i]==0:
               if condPBType4[i-1]==1 and condTA[i]==1:
                    condPBType4.append(1)
               else:
                    condPBType4.append(0)
          else:
               condPBType4.append(float('nan'))
     return condPBType4
