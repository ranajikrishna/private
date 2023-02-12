
import numpy as np
import pandas as pd

import matplotlib 
#matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.metrics import roc_auc_score, auc
from itertools import product

import sys
import pdb


def plot(df):
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig = plt.figure(figsize=(12, 12))

    # Generate a custom diverging colormap
#    cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
#    # Draw the heatmap with the mask and correct aspect ratio
#    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    col_bar = ['zero_bal_upb','int_upb','act_upb','upb','msa']
    pdb.set_trace()



def plot_auc(plt,tprs,base_fpr,aucs):
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    mean_auc = auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

#    fig = Figure(figsize=(4,4))
    plt.plot(base_fpr, mean_tprs, 'b', alpha = 0.8, \
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'blue', alpha = 0.2)
    plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', \
                                                    label = 'Luck', alpha= 0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.grid()
    #plt.axes().set_aspect('equal', 'datalim')
    pdb.set_trace()
    plt.show()

    return


def repeatedKFold_plot(result):

    tprs = []
    aucs = []
    n_split = np.shape(result)[0]
    n_repeat = np.shape(result)[2]
    base_fpr = np.linspace(0, 1, 101)
    combs = list(product(list(range(n_split)),list(range(n_repeat))))
    for i,j in combs:
        plt.plot(base_fpr, result[i][1][j], alpha=0.3) 
#                                    label="ROC fold {}".format(j,i),alpha=0.3)
        tprs.append(result[i][1][j])
        aucs.append(result[i][0][j])

    mean_tprs = np.mean(tprs,axis=0)
    std_tprs = np.std(tprs,axis=0)
    tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
    tprs_lower = mean_tprs - std_tprs

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(base_fpr, mean_tprs, 'b', alpha = 0.8, \
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),)
    plt.fill_between(base_fpr,tprs_lower,tprs_upper,color='blue',alpha = 0.2)
    plt.plot([0, 1], [0, 1],linestyle = '--',lw = 2,color='r', \
                                                    label = 'Luck', alpha= 0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.grid()
    #plt.axes().set_aspect('equal', 'datalim')
    pdb.set_trace()
    plt.show()
    return 
