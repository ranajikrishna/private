
import pdb
import sys
import pandas as pd
import numpy as np
import itertools 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from scipy import interp
import matplotlib.pyplot as plt

import forecast_fft as fcst_fft


def roc_sim(data,k,up):

    col = 'wave_height_51201h'
    frac_list = [0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]
    fcst_prd = [1,3,6,9,12]
    dict_keys = list(itertools.product(frac_list,fcst_prd))
    test_year = pd.Timestamp('2016-01-01')
    test_range = pd.Series(pd.date_range(test_year, periods=24, freq='M'))
    summary_roc_prc = dict.fromkeys(dict_keys)
    itr_list = list(itertools.product(frac_list,fcst_prd))
    for frac, prd in  itr_list:
        tpr,fpr,tnr,fnr = [],[],[],[]
        test = pd.DataFrame([])
        for test_mth in test_range:
            res = fcst_fft.reconstruct_signal(data,frac,test_month=test_mth,fcst_prd=prd,plot=True)
            test = test.append(res.loc[test_mth-pd.DateOffset(months=1):test_mth][1:])

        test=res[test_mth-pd.DateOffset(months=1):]
        test['surf'] = 0
        test.loc[test[col]>=3,'surf'] = 1
        # Receiver Operating Characteristic curve
        fpr, tpr, thr = metrics.roc_curve(test['surf'],test['restored'],pos_label=1)
        roc_stat = pd.DataFrame({'fpr':fpr,'tpr':tpr,'thr':thr})
        base_fpr = np.linspace(0, 1, len(tpr))
        base_tpr = interp(base_fpr, fpr, tpr)
        roc_auc = roc_auc_score(test['surf'],test['restored'],average=None)
        # calculate model precision-recall curve
        pre, rec, th = precision_recall_curve(test['surf'],test['restored'],pos_label=1)
        prc_auc = auc(test['surf'],test['restored'])
#        prc_stat = pd.DataFrame({'pre':pre,'rec':rec,'thr':th})
        summary_roc_prc[(frac,prd)]={'roc': roc_auc}

    pdb.set_trace()
    return 

