
import sys
import pdb

import random


def rand7() -> int:
    '''
    API: rand7(); this fxn. is provided.
    '''
    import numpy as np
    return int(np.random.uniform(1,8))

def rand10():
    '''
    Given the API rand7() that generates a uniform random integer in the 
    range [1, 7], write a function rand10() that generates a uniform random 
    integer in the range [1, 10]. You can only call the API rand7(), and you 
    shouldn't call any other API. Please do not use a language's built-in 
    random API.
    '''

    row = rand7()
    col = rand7()
    # NOTE: `val != col * row` because some numbers will be generated
    # more frequently than other, e.g. 6: (1,6),(2,3),(3,2),(6,1) VS 3: (1,3),
    # (3,1). Similarly, `val != col + row` because 10: (1,9),(2,8),(3,7) etc. VS
    # 3: (1,2),(2,1). Also, `val != row + col * 7` because 1 cannot be generated.
    
    val = row + (col -1 ) * 7

    if (val <= 40):
        return  1 + (val - 1) % 10
    
    col = rand7()
    val1 = (val - 40) + (col - 1) * 7
    if (val1 <= 60):
        return  1 + (val - 1) % 10
    
    col = rand7()
    val2 = (val1 - 60) + (col - 1 ) * 7
    if (val2 <= 20):
        return  1 + (val - 1) % 10
    
    return 1

def main():

    tmp = []    
    tmp2 = []
    for i in range(100000):
        tmp.append(rand10())
        tmp2.append(rand7())

    import matplotlib.pyplot as plt
    plt.hist(tmp,10)
    pdb.set_trace()


    return

if __name__ == '__main__':
    status = main()
    sys.exit()
