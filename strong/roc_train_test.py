
import pdb
import sys
import pandas as pd
import numpy as np
import itertools 
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from scipy import interp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import forecast_fft as fcst_fft

def event_pop(data):

    col = 'restored'
    avg = np.mean(data[col])
    data['restored_mean'] = avg
    return data

def roc_sim(data,k,up):

    col = 'wave_height_51201h'
    #data[col] = data[col] * (data.surf_day)/7
#    col = 'surf_count'
    frac_list =[0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]
#    frac_list =[0.1,0.15,0.2,0.25,0.3,0.35]
    frac_list = [0.07]
    test_mth = pd.Timestamp('2016-01-31')
    plt.style.use('seaborn')
    pdf = PdfPages('./roc_curve_1.pdf')
    summary_auc = dict.fromkeys(frac_list)
    for frac in frac_list:
        tpr,fpr,tnr,fnr = [],[],[],[]
        test = pd.DataFrame([])
        res,per_harm = fcst_fft.reconstruct_signal(data,frac,test_month=test_mth,plot=True)
        test = test.append(res.loc[test_mth-pd.DateOffset(months=1):][1:])
#        test = test.groupby([pd.Grouper(key='date',freq='W-MON')]).apply(event_pop)
        test['surf'] = 0
        test.loc[test['surf_day']>=1,'surf'] = 1
#        test.loc[test['surf_count']>=1,'surf'] = 1

        # Receiver Operating Characteristic curve
        fpr, tpr, thr = metrics.roc_curve(test['surf'],test['restored'],pos_label=1)
        roc_stat = pd.DataFrame({'fpr':fpr,'tpr':tpr,'thr':thr})
        base_fpr = np.linspace(0, 1, len(tpr))
        base_tpr = interp(base_fpr, fpr, tpr)
        roc_auc = roc_auc_score(test['surf'],test['restored'],average=None)
        # calculate model precision-recall curve
        pre, rec, th = precision_recall_curve(test['surf'],test['restored'],pos_label=1)
        prc_auc = auc(rec,pre)
        prc_stat = pd.DataFrame({'pre':pre,'rec':rec})
        summary_auc[frac]={'roc': roc_auc, 'prc': prc_auc}
        fig, axs = plt.subplots(2,tight_layout=True)
        axs[0].plot(base_fpr, base_tpr, label='Harm. pc.: ' + str(round(frac*100,3)) + ' %')
        axs[0].set_ylabel('True positive rate')
        axs[0].set_xlabel('False positive rate')
        axs[0].grid(which='major',color='white',linewidth=0.5)
        axs[0].legend()
        axs[0].set_title('Receiver Operating Characteristic curve. AUC = ' + str(round(roc_auc,3)))
        axs[1].plot(rec, pre, label='Harm. pc.: ' + str(round(frac*100,3)) + ' %')
        axs[1].set_ylabel('Precision')
        axs[1].set_xlabel('Recall/True positive rate')
        axs[1].grid(which='major',color='white',linewidth=0.5)
        axs[1].legend()
        axs[1].set_title('Precision-Recall curve. AUC = ' + str(round(prc_auc,3)))
        test['pred_surf']=0
        test.loc[test.restored>=0.423695668616361,'pred_surf']=1  

#        pdf.savefig(fig)

    pdf.close()
    pdb.set_trace()
    return 

