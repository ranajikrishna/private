
import pdb
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates

from itertools import product

def interpolate_seasonal(data, col_list, plot=False):

    col = col_list[0]
    intp_data = pd.DataFrame({col:data[col]},columns=[col,'yr_mean'])
    index_null = intp_data[intp_data[col].isna()].index
    yr_min = min(intp_data.index).year
    yr_max = max(intp_data.index).year
    period = 1
    yr_range = np.arange(yr_min,yr_max+1,period)

    for dte in index_null:
        
        yr_delta = set(yr_range - dte.year) - {0}
        index_mean=[]
        index_mean1=[]
        for delta in yr_delta:
            index_mean.append(dte + pd.Timedelta(delta * 365, unit='D'))

        intp_data.at[dte,col] = intp_data.loc[index_mean].mean()[col]
        intp_data.at[dte,'yr_mean'] = intp_data.loc[index_mean].mean()[col]

    if plot:
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2,2, sharey=True, tight_layout=True)
        for i in product(range(2), repeat=2):
            axs[i].set_ylabel('Wave hgt.')
            axs[i].set_xlabel('Date')
            axs[i].grid(which='major',color='black',linewidth=0.1)
            
        axs[0,0].plot(data[col])
        axs[0,1].plot(data.loc[(data.index>='2016-06-01') & (data.index<'2017-01-01')][col])
        axs[1,0].plot(intp_data)
        axs[1,1].plot(intp_data.loc[(intp_data.index>='2016-06-01') & (intp_data.index<'2017-01-01')])
        fig.suptitle(col)

    return intp_data 



