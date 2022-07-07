
import pdb
import sys
import pandas as pd
import numpy as np

import forecast_fft as fcst_fft 


def simulate(data, k, up):
    
    summary = pd.DataFrame({'frac': np.linspace(1/k,up,k)}, columns=['frac','mae','max','std'])
    frac_list = [0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]
    fcst_prd = [1,3,6,9,12]
    train_pc = 0.98
    n = int(data.shape[0] * train_pc)
    pdb.set_trace()
    for frac in np.linspace(1/k,up,k):
        res = fcst_fft.reconstruct_signal(data, frac, train_pc, plot=True)
        summary.at[summary.frac==frac,'mae'] = np.mean(abs(res[n:]['error']))
        summary.at[summary.frac==frac,'max'] = max(abs(res[n:]['error']))
        summary.at[summary.frac==frac,'std'] = np.std(res[n:]['error'])
        pdb.set_trace()

    pdb.set_trace()


    return 
