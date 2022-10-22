#conda activate qntdev

import pdb
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import xarray as xr

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata

import hmm_states as hs
import plot_olhcv as po
import getData_stocks as gds
import cointegrated_pairs as cp
import plot_pairs as pp
import analyse_pair as ap
import strategy as st
import compute_returns as cr
import compute_maxdrw as cm

def pctChange(price,freq):
    return [price[i+freq-1]/price[i] -1 for i in range(0,len(price)-freq+1)]

def main():
    start_date = '2010-08-01'
    end_date = '2022-08-01'
    data_df = gds.getData_stocks(start_date,end_date)
    data_df = np.log(data_df)
    search_pairs = False
    if search_pairs:
        pairs = cp.cointegrated_pairs(data_df)
#        pp.plot_pairs(pairs,data_df)

    sel_pair = [('NAS:ADBE','NAS:AMZN'),('NAS:ADSK','NAS:AMZN'),('NAS:ADSK','NAS:ANSS'),
            ('NAS:BKNG','NAS:CHKP'),('NAS:APPL','NAS:ODFL'),('NAS:WTW','NAS:XEL'),
            ('NAS:WTW','NYS:QGEN'), ('NAS:NTES','NYS:QGEN'),('NAS:BKNG','NAS:CSCO'),
            ('NAS:ANSS','NAS:WTW')]

    idx = 8
#    ap.analyse_pair(data_df,sel_pair,idx)
    pair_df = st.strat_one(data_df,sel_pair[idx]) 
    ret_df = cr.compute_returns(pair_df)
    df = cm.compute_maxdrw(ret_df)
    return 


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    status = main()
    sys.exit()
    
    #from quantiacsToolbox.quantiacsToolbox import runts, optimize

    results = runts(__file__)
    #optimize(__file__)

