
import pdb
import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def interpolate_rollmean(data, col_list):

    col = col_list[0]
    intp_data = pd.DataFrame({col:data[col]}, columns = [col,'roll_mean'])
    win = 120
    diff = pd.Timedelta(win, unit='D')
    for dte in intp_data.index[win:]:
        roll_mean = np.mean(intp_data.loc[(intp_data.index>=dte-diff) & (intp_data.index<dte)][col])
        intp_data.at[dte,'roll_mean'] = roll_mean
        if np.isnan(intp_data.loc[dte,col]):
            intp_data.at[dte,col] = roll_mean

    plt.plot(intp_data)
    pdb.set_trace()

    return 
