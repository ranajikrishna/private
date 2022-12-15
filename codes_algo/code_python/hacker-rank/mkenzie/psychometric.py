import pdb
import sys

import numpy as np

def jobOffers(scores,lowerLimits,upperLimits):
    limit_list = [*zip(lowerLimits,upperLimits)]
    scr = np.array(scores)
    for lmt in limit_list:
        print(sum(scr[scr>=min(lmt)]<=max(lmt)))
    return 

def main():

    scr = [1,3,5,6,8]
    low= [2]
    up = [6]
    jobOffers(scr,low,up)

    scr = [4,8,7]
    up=[2,4]
    low=[8,4]
    
    jobOffers(scr,low,up)

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()

