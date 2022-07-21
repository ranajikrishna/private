
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

def roc_sim(data,k,up,col='wave_height_51201h',component=False,plot=False):

#    col = 'wave_height_51201h'
    #data[col] = data[col] * (data.surf_day)/7
#    col = 'surf_count'
    frac_list =[0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]
#    frac_list =[0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19]
    frac_list = list(itertools.product(frac_list,repeat=3))
#    frac_list=[(0.4,0.1,0.2)]
    test_mth = pd.Timestamp('2017-01-01')
    plt.style.use('seaborn')
    pdf = PdfPages('./roc_curve_1.pdf')
    summary_auc = dict.fromkeys(frac_list)
    for frac_trd,frac_sea,frac_red in frac_list:
        tpr,fpr,tnr,fnr = [],[],[],[]
        test = pd.DataFrame([])
        if not component:
            res,per_harm = fcst_fft.reconstruct_signal(data,frac,test_month=test_mth,plot=True)
        else:
            data_trd = pd.DataFrame(data[1].trend).rename(columns={'trend':col}).dropna()
            data_sea = pd.DataFrame(data[1].seasonal).rename(columns={'season':col})
            data_red = pd.DataFrame(data[1].resid).rename(columns={'resid':col})
            data_sea = data_sea.loc[data_trd.index]
            data_red = data_red.loc[data_trd.index]
            res_trd, per_harm = fcst_fft.reconstruct_signal(data_trd,frac_trd,test_month=test_mth,plot=False)
            res_sea, per_harm = fcst_fft.reconstruct_signal(data_sea,frac_sea,test_month=test_mth,plot=False)
#            res_red, per_harm = fcst_fft.reconstruct_signal(data_red,frac,test_month=test_mth,plot=True)
            res = res_trd + res_sea
            res[col] += data[1].resid   # Add residual to the `col` (and not to `reconstruct` to prevent leakage.)
#            res = res_trd * res_sea
#            res[col] = res[col] * data[1].resid
            res['surf_day'] = data[0]['surf_day']
            res['surf_bit'] = np.ceil(res.surf_day/8)
            #res['good_surf'] = res['error']*res['surf_bit']
#            data_red[col] = data_red[col]*res['surf_bit']
           # res.loc[res['good_surf'] <0,'good_surf']=0
#            data_red.loc[data_red[col] <0,col]=0
           # data_surf = pd.DataFrame(res['good_surf'],columns=['good_surf'])
           # res_good, per_harm = fcst_fft.reconstruct_signal(data_surf,0.8,test_month=test_mth,col='good_surf',plot=True)
            res_red, per_harm = fcst_fft.reconstruct_signal(data_red,frac_red,test_month=test_mth,plot=False)
           # res['restored'] += res_good['restored']
            res['restored'] += res_red['restored']
            res['error'] = res[col] - res['restored'] 
        #test = test.append(res.loc[test_mth-pd.DateOffset(months=1):][1:])
        test = test.append(res.loc[test_mth:])
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
        #summary_auc[frac]={'roc': roc_auc, 'prc': prc_auc}
        summary_auc[(frac_trd,frac_sea,frac_red)]={'roc': roc_auc, 'prc': prc_auc}

        if plot:
            # ==== Plot ROC/PRC Curves ====
            fig, axs = plt.subplots(2,tight_layout=True)
            axs[0].plot(base_fpr, base_tpr)
            axs[0].set_ylabel('True positive rate')
            axs[0].set_xlabel('False positive rate')
            axs[0].grid(which='major',color='white',linewidth=0.5)
            axs[0].legend()
            axs[0].set_title('Receiver Operating Characteristic curve. AUC = ' + str(round(roc_auc,3)))
            axs[1].plot(rec, pre)
            axs[1].set_ylabel('Precision')
            axs[1].set_xlabel('Recall/True positive rate')
            axs[1].grid(which='major',color='white',linewidth=0.5)
            axs[1].legend()
            axs[1].set_title('Precision-Recall curve. AUC = ' + str(round(prc_auc,3)))
            test['pred_surf']=0
            test.loc[test.restored>=0.423695668616361,'pred_surf']=1  

            # ==== Plot ROC/PRC Curves ====
            fig, axs = plt.subplots(2,tight_layout=True)
            test[[col,'restored']].plot(ax=axs[0])
            axs[0].set_ylabel('Wave height')
            axs[0].grid(which='major',color='white',linewidth=0.5)
            axs[0].xaxis.label.set_visible(False)
            axs[0].legend(['True wave height','Estimated wave height'],loc='upper right')
            test['error'].plot(ax=axs[1])
            axs[1].set_ylabel('Error')
            axs[1].set_xlabel('Date')
            axs[1].grid(which='major',color='white',linewidth=0.5)
            fig.suptitle('True and Estimated wave heights, and Error')

#        pdf.savefig(fig)

    pdf.close()
#    pd.DataFrame(summary_auc).to_excel("summary_auc_xxxxxx.xlsx")
    pdb.set_trace()
        
    return 

