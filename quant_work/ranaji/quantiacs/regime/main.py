
import pdb
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import xarray as xr

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata

#from statsmodels import regression
from hmmlearn import hmm
from hmmlearn.base import ConvergenceMonitor

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

def hmm_states(series,num_cmp,plot=False):
    remodel = hmm.GaussianHMM(n_components=num_cmp, covariance_type="full", n_iter=100, verbose=True)
    remodel.fit(np.array(series).reshape(-1,1))
    if plot:
        # Plot `log_probability` to monitor `convergence`.
        plt.figure()
        plt.plot(remodel.monitor_.history)
        plt.xlabel('iteration'); plt.ylabel('log_probability'); plt.grid()
    # Determine states from HMM.
    state = remodel.predict(np.array(series).reshape(-1,1))
    return state

def pctChange(price,freq):
    return [price[i+freq-1]/price[i] -1 for i in range(0,len(price)-freq+1)]


def getData(asset_tkr,start_date,end_date):

    data = qndata.futures.load_data(assets=[asset_tkr], min_date=start_date, \
                        max_date=end_date, dims=("field", "time", "asset"))

    data_df = pd.DataFrame({'open':data.sel(field='open').data.ravel(),
                  'close': data.sel(field='close').data.ravel(),
                  'high': data.sel(field='high').data.ravel(),
                  'low': data.sel(field='low').data.ravel(),
                  'vol_day': data.sel(field='vol').data.ravel(),
                  'open_int': data.sel(field='oi').data.ravel(),
                  'ctr_roll_ovr': data.sel(field='roll').data.ravel()
                  }, index=data.time.data)
    
    return data_df 

def main():
    asset_tkr = 'F_YM' # SPDR Dow Jones Industrial Average ETF
    start_date = '2010-08-01'
    end_date = '2022-08-01'
    data_df = getData(asset_tkr, start_date,end_date)
    #data_df['open_var'] = data_df['open'].rolling(14).var()
    data_df['open_var'] = data_df['vol_day'].rolling(14).var()
    freq = 2
    
    for ind,price in enumerate(['open']): #'close','high','low','open_var']):
        component = ['red','green','blue']
        component = ['red','green']
        df = pd.DataFrame({'ret_'+price:pctChange(data_df[price],freq)},\
                                                   index=data_df.index[1:])
        df.dropna(inplace=True)
        df['hid_ste_'+str(price)] = hmm_states(df['ret_'+price], \
                                                            len(component))
        data_df = data_df.join(df)
        df = pd.DataFrame({price+'_var':data_df[price+'_var']})
        df.dropna(inplace=True)
        component = ['red','green']
        df['hid_ste_'+str(price)+'_var'] = hmm_states(df[price+'_var'],len(component))
        data_df = data_df.join(df['hid_ste_'+str(price)+'_var'])
        component = ['red','green','blue']
        plot_olhcv(data_df,price,component)

    pdb.set_trace()

    return 


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    status = main()
    sys.exit()
    
    #from quantiacsToolbox.quantiacsToolbox import runts, optimize

    results = runts(__file__)
    #optimize(__file__)




