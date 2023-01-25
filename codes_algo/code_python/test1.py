
import sys
import pdb
import numpy as np
import pandas as pd
import datetime,time
from sklearn.linear_model import LinearRegression
import re


#def sum(x,y,r):
#
#    for i in x:
#        for j in y:


def main():
    arr = int(1,-2,3,2,1;1,2,4,-8,2;2,1,6,4,3;3,-7,1,0,-4;4,3,2,2,1)
    pdb.set_trace()

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()

#    print((3%6) + 4)
#    print(((3%6)+5)/2)
#    print((3%6)-2)
#    print((3%6)+2)



for line in sys.stdin:
    list_rows = line.split(';')
    mat = []
    for row in list_rows:
        list_val = row.split(',')
        arr = []
        for val in list_val:
            arr.append(val)
        mat.append(arr)
           
