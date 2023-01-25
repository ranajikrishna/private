
'''   
NOTE: This algo. computes ustomer lifetime value using a Bayesian Framework.
For reference: https://towardsdatascience.com/bayesian-customer-lifetime-values-modeling-using-pymc3-d770676f5c06
(or bayesian_customer_lifetime_values_modeling_using_PyMC3.pdf)

It uses MCMC algorithm in its computation. 
'''

import sys
import pdb
import pandas as pd

from lifetimes.datasets import load_dataset
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.utils import calibration_and_holdout_data

from pymc3.math import log, exp, where
import pymc3 as pm
import numpy as np

def get_rfm(transactions):
    rfm = summary_data_from_transaction_data(transactions=transactions,
                                         customer_id_col='customer_id',
                                         datetime_col='date',
                                         monetary_value_col = 'amount',
                                         observation_period_end=pd.to_datetime('1998-06-30'),
                                         freq='W')
    return rfm

def get_holdout(transactions):
    rfm_cal_holdout = calibration_and_holdout_data(transactions=transactions,
                                               customer_id_col='customer_id', 
                                               datetime_col='date',
                                               monetary_value_col = 'amount',
                                               freq='W',
                                               calibration_period_end='1998-01-01',
                                               observation_period_end='1998-06-30' )
    return rfm_cal_holdout

def get_data():
    transactions = load_dataset(
        filename='CDNOW_sample.txt', 
        header=None, 
        delim_whitespace=True, 
        names=['customer_id', 'customer_index', 'date', 'quantity', 'amount'],
        converters={'date': lambda x: pd.to_datetime(x, format="%Y%m%d")}
    )
    return transactions

def model(rfm_cal_holdout):

    # We use the "calibration" portion of the dataset to train the model
    N = rfm_cal_holdout.shape[0] # number of customers
    x = rfm_cal_holdout['frequency_cal'].values # repeat purchase frequency
    t_x = rfm_cal_holdout['recency_cal'].values # recency
    T = rfm_cal_holdout['T_cal'].values # time since first purchase (T)

    # Modeling step
    bgnbd_model = pm.Model()
    with bgnbd_model:
        
        # Priors for r and alpha, the two Gamma parameters
        r = pm.TruncatedNormal('r', mu=8, sigma=7, lower=0, upper=40)
        alpha = pm.TruncatedNormal('alpha', mu=0.5, sigma=5, lower=0, upper=10)

        # Priors for a and b, the two Beta parameters
        a = pm.TruncatedNormal('a', mu=1, sigma=5, lower=0, upper=10)
        b = pm.TruncatedNormal('b', mu=1, sigma=5, lower=0, upper=10)

        # lambda_ (purchase rate) is modeled by Gamma, which is a child distribution of r and alpha
        lambda_ = pm.Gamma('lambda', alpha=r, beta=alpha, shape=N, testval=np.random.rand(N))
        
        # p (dropout probability) is modeled by Beta, which is a child distribution of a and b
        p = pm.Beta('p', alpha=a, beta=b, shape=N, testval=np.random.rand(N))
        
        def logp(x, t_x, T):
            """
            Loglikelihood function
            """    
            delta_x = where(x>0, 1, 0)
            A1 = x*log(1-p) + x*log(lambda_) - lambda_*T
            A2 = (log(p) + (x-1)*log(1-p) + x*log(lambda_) - lambda_*t_x)
            A3 = log(exp(A1) + delta_x * exp(A2))
            return A3
        
        # Custom distribution for BG-NBD likelihood function
        loglikelihood = pm.DensityDist("loglikelihood", logp, observed={'x': x, 't_x': t_x, 'T': T})

    # Sampling step
    SEED = 8 
    SAMPLE_KWARGS = {
        'chains': 1,
        'draws': 4000,
        'tune': 1000,
        'target_accept': 0.7,
        'random_seed': [
            SEED,
        ]
    }
    with bgnbd_model:
        trace = pm.sample(**SAMPLE_KWARGS)
        
    # It's a good practice to burn (discard) early samples
    # these are likely to be obtained before convergence
    # they aren't representative of our posteriors.
    trace_trunc = trace[3000:]

    return




def main():
    data = get_data()
    rfm = get_rfm(data)
    hold_out = get_holdout(data)
    model(hold_out)

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
