
import pdb
import sys
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

import forecast_fft as fcst_fft 


def simulate(data, k, up):

    col = 'wave_height_51201h' 
    #summary = pd.DataFrame({'frac': np.linspace(1/k,up,k)}, columns=['frac','mae','max','std'])
    frac_list = [0.001]#0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]
    fcst_prd = [1]#,3,6,9,12]
    metrics = ['mae','max_','std_dev','tpr','fpr','tnr','fnr']
    #summary = dict.fromkeys(frac_list,dict.fromkeys(fcst_prd,dict.fromkeys(metrics)))
    dict_keys = list(itertools.product(frac_list,fcst_prd))
    summary_mae = dict.fromkeys(dict_keys)
    summary_max = dict.fromkeys(dict_keys)
    summary_std_dev = dict.fromkeys(dict_keys)
    summary_tpr = dict.fromkeys(dict_keys)
    summary_fpr = dict.fromkeys(dict_keys)
    summary_tnr = dict.fromkeys(dict_keys)
    summary_fnr = dict.fromkeys(dict_keys)
    test_year = pd.Timestamp('2016-01-01')
    test_range = pd.Series(pd.date_range(test_year, periods=24, freq='M'))
    itr_list = list(itertools.product(frac_list,fcst_prd))
    for frac, prd in  itr_list:
        mae,max_,std_dev =  [],[],[]
        tpr,fpr,tnr,fnr = [],[],[],[]
        for test_mth in test_range:
            res = fcst_fft.reconstruct_signal(data, frac,test_month=test_mth,fcst_prd=prd, plot=True)
            test = res.loc[test_mth-pd.DateOffset(months=prd):test_mth][1:]
            test['tp'] = (test[col] >= 3) & (test['restored'] >= 3)
            test['fp'] = (test[col] < 3) & (test['restored'] >= 3)
            test['tn'] = (test[col] < 3) & (test['restored'] < 3)
            test['fn'] = (test[col] >= 3) & (test['restored'] < 3)
            mae.append(np.mean(abs(test['error'])))
            max_.append(max(abs(test['error'])))
            std_dev.append(np.std(test['error']))
            tpr.append(sum(test[col]>=3) and sum(test.tp)/sum(test[col]>=3) or 0)
            fpr.append(sum(test[col]<3) and sum(test.fp)/sum(test[col]<3) or 0)
            tnr.append(sum(test[col]<3) and sum(test.tn)/sum(test[col]<3) or 0)
            fnr.append(sum(test[col]>=3) and sum(test.fn)/sum(test[col]>=3) or 0)

        pdb.set_trace()
        summary_mae[(frac,prd)] = np.mean(mae)
        summary_max[(frac,prd)] = np.mean(max_)
        summary_std_dev[(frac,prd)] = np.mean(std_dev)
        summary_tpr[(frac,prd)] = np.mean(tpr)
        summary_fpr[(frac,prd)] = np.mean(fpr)
        summary_tnr[(frac,prd)] = np.mean(tnr)
        summary_fnr[(frac,prd)] = np.mean(fnr)

    pdb.set_trace()
    print(pd.Series(summary_mae).unstack(fill_value=0))


    return 
