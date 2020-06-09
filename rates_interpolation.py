#calculated interpolated rates for each period of time. List of interpolated rates is produced. Later this will go to loop for each country (file First.py)
import pandas as pd
#%%
def interpolate_rates(numOfCountry, DF1W, DF1M, DF3M, daysToExer):
     '''Calculated interpolated rates for tenors 1 week, 1 month, 3 months (5 days, 21 and 64 days). 
     :inputs: lists rates in percent per year (eg. 12.5)
     :return: list with interpolated rates 
     '''
     countryRatesDF = pd.DataFrame()
     countryRatesDF['1W'] = list(DF1W.iloc[:,numOfCountry]) 
     countryRatesDF['1M'] = list(DF1M.iloc[:,numOfCountry]) 
     countryRatesDF['3M'] = list(DF3M.iloc[:,numOfCountry]) 
     countryRatesDF.index = DF1W.index
#list with description of available rates for interpolation 
     rateType = [None]*len(countryRatesDF)
     for i in range(0, len(countryRatesDF)):
          if pd.notna(countryRatesDF.iloc[i,0]) and pd.notna(countryRatesDF.iloc[i,1]) and pd.notna(countryRatesDF.iloc[i,2]):
               rateType[i] = 'All'
     
          elif pd.notna(countryRatesDF.iloc[i,0]) and pd.isna(countryRatesDF.iloc[i,1]) and pd.isna(countryRatesDF.iloc[i,2]):
               rateType[i] = 'One1'
          elif pd.isna(countryRatesDF.iloc[i,0]) and pd.notna(countryRatesDF.iloc[i,1]) and pd.isna(countryRatesDF.iloc[i,2]):
               rateType[i] = 'One2'
          elif pd.isna(countryRatesDF.iloc[i,0]) and pd.isna(countryRatesDF.iloc[i,1]) and pd.notna(countryRatesDF.iloc[i,2]):
               rateType[i] = 'One3'
          
          elif pd.notna(countryRatesDF.iloc[i,0]) and pd.notna(countryRatesDF.iloc[i,1]) and pd.isna(countryRatesDF.iloc[i,2]):
               rateType[i] = 'Two12'
          elif pd.notna(countryRatesDF.iloc[i,0]) and pd.isna(countryRatesDF.iloc[i,1]) and pd.notna(countryRatesDF.iloc[i,2]):
               rateType[i] = 'Two13'
          elif pd.isna(countryRatesDF.iloc[i,0]) and pd.notna(countryRatesDF.iloc[i,1]) and pd.notna(countryRatesDF.iloc[i,2]):
               rateType[i] = 'Two23'
     
          else:
               rateType[i] = 'None'

# rates interpolation technique based on rateType list
     interpolRates = [None]*len(countryRatesDF)

     for i in range(0, len(countryRatesDF)):
          if daysToExer[i]>=64:
               if any(item in rateType[i] for item in ['All','Two23','Two13','One3']):
                    interpolRates[i] = countryRatesDF.iloc[i,2]
               elif any(item in rateType[i] for item in ['Two12','One2']):
                    interpolRates[i] = countryRatesDF.iloc[i,1]
               elif any(item in rateType[i] for item in ['One1']):
                    interpolRates[i] = countryRatesDF.iloc[i,0]
               else:
                    interpolRates[i] = float('nan')
     
          elif daysToExer[i]>=21 and daysToExer[i]<64:
               if any(item in rateType[i] for item in ['All','Two23']):
                    interpolRates[i] = (64-daysToExer[i])/(64-21)*countryRatesDF.iloc[i,1]+(daysToExer[i]-21)/(64-21)*countryRatesDF.iloc[i,2]
               elif any(item in rateType[i] for item in ['Two13']):
                    interpolRates[i] = (64-daysToExer[i])/(64-5)*countryRatesDF.iloc[i,0]+(daysToExer[i]-5)/(64-5)*countryRatesDF.iloc[i,2]
               elif any(item in rateType[i] for item in ['Two12','One2']):
                    interpolRates[i] = countryRatesDF.iloc[i,1]
               elif any(item in rateType[i] for item in ['One1']):
                    interpolRates[i] = countryRatesDF.iloc[i,0]
               elif any(item in rateType[i] for item in ['One3']):
                    interpolRates[i] = countryRatesDF.iloc[i,2]
               else:
                    interpolRates[i] = float('nan')
     
          elif daysToExer[i]>=5 and daysToExer[i]<21:
               if any(item in rateType[i] for item in ['All','Two12']):
                    interpolRates[i] = (21-daysToExer[i])/(21-5)*countryRatesDF.iloc[i,0]+(daysToExer[i]-5)/(21-5)*countryRatesDF.iloc[i,1]
               elif any(item in rateType[i] for item in ['Two23','One2']):
                    interpolRates[i] = countryRatesDF.iloc[i,1]
               elif any(item in rateType[i] for item in ['One3']):
                    interpolRates[i] = countryRatesDF.iloc[i,2]
               elif any(item in rateType[i] for item in ['One1']):
                    interpolRates[i] = countryRatesDF.iloc[i,0]
               elif any(item in rateType[i] for item in ['Two13']):
                    interpolRates[i] = (64-daysToExer[i])/(64-5)*countryRatesDF.iloc[i,0]+(daysToExer[i]-5)/(64-5)*countryRatesDF.iloc[i,2]
               else:
                    interpolRates[i] = float('nan')
               
          else:
               if any(item in rateType[i] for item in ['All','Two13','Two12','One1']):
                    interpolRates[i] = countryRatesDF.iloc[i,0]
               elif any(item in rateType[i] for item in ['Two23','One2']):
                    interpolRates[i] = countryRatesDF.iloc[i,1]
               elif any(item in rateType[i] for item in ['One3']):
                    interpolRates[i] = countryRatesDF.iloc[i,2]
               else:
                    interpolRates[i] = float('nan')
     return interpolRates
