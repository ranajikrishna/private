
import pdb
import sys
import numpy as np
import datetime,time
from sklearn.linear_model import LinearRegression


def predictTemperature(startDate, endDate, temperature, n):
    tmp = ['2013-01-01 00:00','2013-01-01 01:00','2013-01-01 02:00']

    p = int(len(temperature)/24)
    x = []
    for i in range(1,((24*p)+1)):
        x.append(i)
    y = temperature
    lm = LinearRegression()
    lm.fit(np.asarray(x).reshape(-1,1),y)
    
    f = x[-1]+1
    z = []
    for i in range(24*n):
        z.append(f)
        f += 1
    return(lm.predict(np.asarray(z).reshape(-1,1)).tolist())

# call the utility function
#print(predictTemperature(startDate, endDate, data, n))




def main():
    data=[10.0,11.1,12.3,13.2,14.8,15.6,16.7,17.5,18.9,19.7,20.7,21.1,22.6,23.5,24.9,25.1,26.3,27.8,28.8,29.6,30.2,31.6,32.1,33.7]
    startDate = '2013-01-01'
    endDate = '2013-01-01'
    p = 1
    n = 1

    tmp = predictTemperature(startDate, endDate, data, n)

    pdb.set_trace()

    return

if __name__ == '__main__':
    status = main()
    sys.exit()


