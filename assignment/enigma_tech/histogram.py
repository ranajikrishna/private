
import sys
import pdb
import random

import math
import collections


def histogram(nums: list([float])) -> dict:
    '''
    Given a list of real numbers, `nums`, create a histogram of distribution
    of n bins without using pre-built libraries. 
    '''
    
    min_num = min(nums)
    max_num = max(nums)
    num_bins = 1000
    bins = []
    j = 0
    bins = collections.defaultdict(int)
    for i in nums:
        # f(x) = ceiling((x-min)/(max-min) * (bins-1))
        # Intuition: min - max -> 1000 steps
        #              i - min -> x steps 
        bins[math.ceil(((i - min_num)/(max_num - min_num)) * (num_bins-1))] += 1
        if j < num_bins:
            bins[j] += 0
            j += 1
    return bins 

def main():
    n = 1000
    nums =[random.uniform(-1.5, 1.9) for _ in range(n)]
    result = histogram(nums)    

    pdb.set_trace()
    return 

if __name__ == '__main__':
    status = main()
    sys.exit()

