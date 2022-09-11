
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_olhcv(data_df,price,component):
    fig, axs = plt.subplots(5,tight_layout=True)
    axs[0].plot(data_df[price])
    axs[0].set_ylabel(price)
    axs[0].set_xlabel('Date')
    axs[0].grid(which='major',color='gray',linewidth=0.3)
    axs[0].set_title('SPDR Dow Jones Industrial Average ETF')
    for ind,clr in enumerate(component):
        sel = data_df['hid_ste_'+price]==ind
        axs[1].scatter(data_df[sel].index,data_df[sel]['ret_'+str(price)],color=clr,s=2)
        axs[1].set_ylabel('Daily Returns'); axs[1].set_title('Returns'); axs[1].grid()
        axs[2].scatter(data_df[sel].index,data_df[sel][price],color=clr,s=2)
        axs[2].set_title('States - Returns');axs[2].grid()
        sel = data_df['hid_ste_'+price+'_var']==ind
        axs[3].scatter(data_df[sel].index,data_df[sel][price+'_var'],color=clr,s=2)
        axs[3].set_title('Daily Volume');axs[3].grid()
        axs[4].scatter(data_df[sel].index,data_df[sel][price],color=clr,s=2)
        axs[4].set_title('States - DailyVolume');axs[4].grid()
    
    pdb.set_trace()
    return

