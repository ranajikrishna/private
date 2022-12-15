
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
import getData_futures as gdf
import getData_stocks as gds

def pctChange(price,freq):
    return [price[i+freq-1]/price[i] -1 for i in range(0,len(price)-freq+1)]

def main():
    asset_tkr = 'F_YM' # SPDR Dow Jones Industrial Average ETF
    start_date = '2010-08-01'
    end_date = '2022-08-01'
    data_df = gdf.getData_futures(asset_tkr, start_date,end_date)
    #data_df['open_var'] = data_df['open'].rolling(14).var()
    data_df['open_var'] = data_df['vol_day'].rolling(14).var()
    freq = 2
    
    for ind,price in enumerate(['open']): #'close','high','low','open_var']):
        component = ['red','green','blue']
        component = ['red','green']
        df = pd.DataFrame({'ret_'+price:pctChange(data_df[price],freq)},\
                                                   index=data_df.index[1:])
        df.dropna(inplace=True)
        df['hid_ste_'+str(price)] = hs.hmm_states(df['ret_'+price], \
                                                            len(component))
        data_df = data_df.join(df)
        df = pd.DataFrame({price+'_var':data_df[price+'_var']})
        df.dropna(inplace=True)
        component = ['red','green']
        df['hid_ste_'+str(price)+'_var'] = hs.hmm_states(df[price+'_var'],len(component))
        data_df = data_df.join(df['hid_ste_'+str(price)+'_var'])
        component = ['red','green','blue']
        po.plot_olhcv(data_df,price,component)

    pdb.set_trace()

    return 


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    status = main()
    sys.exit()
    
    #from quantiacsToolbox.quantiacsToolbox import runts, optimize

    results = runts(__file__)
    #optimize(__file__)

