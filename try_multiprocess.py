import multiprocessing  
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

#%%

aZero = [20, 40]
bZero = [30, 20]

#aVector = [float('nan')]*5
#bVector = [float('nan')]*5

def func1(aInit, bInit):
    print('start func1')
    a = aInit+bInit
    b = aInit-bInit
    time.sleep(1)
    print ('end func1')
    return [a, a/2, a/4], b

def func2(aInit, bInit):
    print('start func2')
    a = 0.5*aInit+0.5*bInit
    b = 0.5*aInit-0.5*bInit
    time.sleep(1)
    print ('end func2')
    return [a, a/2, a/4], b
 
def func3(aInit, bInit):
    print('start func3')
    a = 0.25*(aInit+bInit)
    b = 0.25*(aInit-bInit)
    time.sleep(1)
    print ('end func3')
    return [a, a/2, a/4], b

def func4(aInit, bInit):
    print('start func4')
    a = 0.125*(aInit+bInit)
    b = 0.125*(aInit-bInit)
    time.sleep(1)
    print ('end func4')
    return [a, a/2, a/4], b

def func5(aInit, bInit):
    print('start func5')
    a = 0.0625*(aInit+bInit)
    b = 0.0625*(aInit-bInit)
    time.sleep(1)
    print ('end func5')
    return [a, a/2, a/4], b

vectorToSave = []
valuesToSave = []

def multi_processing(aZero, bZero):
     n_cpus = multiprocessing.cpu_count()
     pool = ProcessPoolExecutor(max_workers=n_cpus)
     futureVector = [float('nan')]*5
     futureVector[0] = pool.submit(func1, aInit=aZero, bInit=bZero)
     futureVector[1] = pool.submit(func2, aInit=aZero, bInit=bZero)
     futureVector[2] = pool.submit(func3, aInit=aZero, bInit=bZero)
     futureVector[3] = pool.submit(func4, aInit=aZero, bInit=bZero)
     futureVector[4] = pool.submit(func5, aInit=aZero, bInit=bZero)
     return futureVector

#start = time.time()

 #maybe should change order for loop and if
if __name__ == '__main__':
     for i in range(0,len(aZero)):
          results = multi_processing(aZero[i],bZero[i])
          vectors = [x.result()[0] for x in results]
          values = [x.result()[1] for x in results]
          #print(results)
          #print(vector[0].done())
          print(vectors[0])
          print(values[0])
          #print([x.result() for x in results])
          vectorToSave.append(vectors[0])
          valuesToSave.append(values[0])
          
     testDF = pd.DataFrame(vectorToSave)
     testDF.to_csv('./testDF.csv', sep=';', decimal=',')


#end = time.time()
#print(f'\nTime to complete: {end - start:.2f}s\n')
