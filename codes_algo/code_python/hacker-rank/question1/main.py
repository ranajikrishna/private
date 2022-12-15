
import pdb
import sys
import numpy as np
import pandas as pd



def calcMissin():

    data =[-5,4,-2,3,1]#,-1,-6,-1,0,5])

    sum_ = data[0]
    tmp = [sum_]
    for item in data[1:]:
        sum_ += item 
        tmp.append(sum_)

    sum_ = (-1 * min(tmp)) + 1
    tmp1 = [sum_]
    for item in data:
        sum_ += item 
        tmp1.append(sum_)

    pdb.set_trace()

    return

if __name__ == '__main__':

#    txt = open('input.txt','r')
#    data = txt.readlines()
#    txt.close()
#
#    readings = list(map(lambda x: x.strip(),data))
#
#    pdb.set_trace()

    calcMissin()

