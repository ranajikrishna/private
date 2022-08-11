
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


def hmm_para_estimation(series):
    remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, verbose=True)
    remodel.fit(np.array(series).reshape(-1,1))
    # Plot `log_probability` to monitor `convergence`.
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
    freq = 2
    open_df = pd.DataFrame({'ret_open':pctChange(data_df.open,freq)},index=data_df.index[1:])
    open_df['hid_ste_open'] = hmm_para_estimation(open_df.ret_open)

    data_df = data_df.join(open_df)
    open_df['hid_ste_open'].hist(bins=3)

    # ==== Plot ROC/PRC Curves ====
    fig, axs = plt.subplots(3,tight_layout=True)
    axs[0].plot(data_df.open)
    axs[0].set_ylabel('Open price')
    axs[0].set_xlabel('Date')
    axs[0].grid(which='major',color='gray',linewidth=0.3)
    axs[0].legend()
    axs[0].set_title('SPDR Dow Jones Industrial Average ETF')
    axs[1].plot(data_df.ret_open)
    axs[1].set_ylabel('Daily Returns')
    axs[1].set_xlabel('Date')
    axs[1].grid(which='major',color='gray',linewidth=0.3)
    axs[1].legend()
    axs[1].set_title('Return of SPDR Dow Jones Industrial Average ETF')
    for ind,clr in enumerate(['red','green','blue']):
        sel = data_df.hid_ste_open==ind
        axs[2].scatter(data_df[sel].index,data_df[sel]['open'],color=clr,s=0.5)
    pdb.set_trace()

    return 


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    status = main()
    sys.exit()
    
    #from quantiacsToolbox.quantiacsToolbox import runts, optimize

    results = runts(__file__)
    #optimize(__file__)




