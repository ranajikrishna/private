
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from hmmlearn import hmm
from hmmlearn.base import ConvergenceMonitor

def hmm_states(series,num_cmp,plot=False):
    remodel = hmm.GaussianHMM(n_components=num_cmp, covariance_type="full", \
                                                        n_iter=100, verbose=True)
    remodel.fit(np.array(series).reshape(-1,1))

    if plot:
        # Plot `log_probability` to monitor `convergence`.
        plt.figure()
        plt.plot(remodel.monitor_.history)
        plt.xlabel('iteration'); plt.ylabel('log_probability'); plt.grid()
    # Determine states from HMM.
    state = remodel.predict(np.array(series).reshape(-1,1))
    return state

