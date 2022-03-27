
import sys
import pdb
import numpy as np
from itertools import permutations


def probabilityOfHeads(prob: list([float]), target: int) -> float:
    '''
    Dynamic programming: we account for the contribution of each coin towards 
    heads and tails. 
    YouTube: https://www.youtube.com/watch?v=jgiZlGzXMBw
    '''
    dp = [1] + [0] * target
    for index, p in enumerate(prob):
        # Given coin with prob. p ...
        pdb.set_trace()
        for k in range(min(index+1,target), -1, -1):
        # ... compute prob. of getting k heads. 
            dp[k] = (dp[k - 1] if k else 0) * p + dp[k] * (1 - p)   
            # i.e. k heads = (k-1 heads and 1 head) or (k heads and 1 tail).
    return dp[target]

def probabilityOfHeads_v2(prob: list([float]), target: int) -> float:
    flips = np.array([1] * target + [0] * (len(prob) - target))
    prb = np.array(prob)
    total_prb = 0
    for perm in  set(permutations(flips)): 
        perm = np.array(perm)
        total_prb += np.product((prb * perm) + (1-prb) * (1-perm))

    return total_prb

def main():
    '''
    You have some coins.  The i-th coin has a probability prob[i] of facing 
    heads when tossed. Return the probability that the number of coins 
    facing heads equals target if you toss every coin exactly once.
    '''

    target = 3
    prob = [0.5, 0.1, 0.4, 0.2, 0.9]
#    prob = [0.5,0.5,0.5,0.5,0.5]
#    prob = [0.4]
    print(probabilityOfHeads(prob,target))

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
