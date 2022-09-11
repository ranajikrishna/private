
import pdb
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyse_pair(data_df,sel_pair,idx):
    pair_df = pd.DataFrame({sel_pair[idx][0]:data_df[sel_pair[idx][0]], \
                                    sel_pair[idx][1]:data_df[sel_pair[idx][1]]})

    pair_df['ratio'] = pair_df[sel_pair[idx][0]]/pair_df[sel_pair[idx][1]]
#    pair_df['zscore'] = (pair_df['ratio'] - \
#                            np.mean(pair_df['ratio']))/np.std(pair_df['ratio'])
    win = 30
    pair_df['zscore'] = (pair_df.iloc[win-1:]['ratio']-pair_df['ratio'].rolling(win).mean())/ \
                                            pair_df['ratio'].rolling(win).std()
    pair_df['delta_z'] = pair_df.zscore.shift() - pair_df.zscore

    # === Plot === 
    fig, ax1 = plt.subplots() 
    ax1.plot(pair_df[sel_pair[idx][0]],label=sel_pair[idx][0]) 
    ax1.plot(pair_df[sel_pair[idx][1]],label=sel_pair[idx][1]) 
    ax1.legend()
    ax1.set_ylabel('Price')
  
    ax2 = ax1.twinx() 
    ax2.plot(pair_df['zscore'], color = 'blue',label=str(win)+'-day ratio')  
    ax2.legend()
    ax2.set_ylabel('Ratio',loc='bottom')

    return 
