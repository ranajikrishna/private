'''
    Testing Hoeffding's Inequality.
'''

from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
import pdb


def flip():
    trial = list()
    [trial.append(bernoulli.rvs(size=10, p=0.5)) for _ in range(0,1000)]
    return trial 

def main():
    c_1 = list()
    c_rand = list()
    c_min = list()
    N = 10000
    eps = np.arange(0.1,0.9,0.01)
    for _ in range(0,N):
        trial = flip()
        np.array([c_1.append(sum(trial[0])/len(trial[0]))])
        np.array([c_rand.append(sum(trial[random.randint(0,999)])/len(trial[0]))])
        np.array([c_min.append(min([*map(sum,trial)])/len(trial[0]))]) 

    c1_lhs = list()
    crand_lhs = list()
    cmin_lhs = list()
    for i in eps:
        c1_lhs.append(sum(abs(np.array(c_1)-0.5)>i)/N)
        crand_lhs.append(sum(abs(np.array(c_rand)-0.5)>i)/N)
        cmin_lhs.append(sum(abs(np.array(c_min)-0.5)>i)/N)
    hoeff = 2*np.exp(-2*eps**2*N)
    data = {'c_1':c1_lhs
            ,'c_rand':crand_lhs
            ,'c_min':cmin_lhs
            ,'hoeff':hoeff
            ,'epsilon':eps
            }
    data = pd.DataFrame(data,columns=['c_1','c_rand','c_min','hoeff','epsilon'])

    plt.figure()
    sns.lineplot(x='epsilon', y='c_1', data=data, color='blue')
    sns.lineplot(x='epsilon', y='c_rand', data=data, color='green')
    sns.lineplot(x='epsilon', y='c_min', data=data, color='red')
    sns.lineplot(x='epsilon', y='hoeff', data=data,color='black')
    pdb.set_trace()
    return 

if __name__ == '__main__':
    status = main()
