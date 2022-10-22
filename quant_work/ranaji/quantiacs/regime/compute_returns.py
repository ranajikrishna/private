
import pdb
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

def compute_returns(data):
    '''
    We use hedge account to compute returns of the pairs trading trategy.
    '''
    asset1, asset2 = data.columns[0], data.columns[1]
    data['nav_long'], data['nav_short'], data['NAV'], data['pos'], data['port_ret'] = 0,0,0,0,0
    nav_long,nav,flag = 1,0,0
    close = True
    for idx,row in data.iterrows():
        data.loc[idx,'nav_long'] += nav_long
        if (row.signal+flag!=0 and not close):
            # Trade continue.
            asset = asset1 if flag==1 else asset2
            ret_long = row[asset] - price_long 
            nav_long = np.exp(ret_long) * nav_long
            data.loc[idx,'nav_long'] = nav_long 
            asset=asset2 if flag==1 else asset1
            ret_short = 1 - np.exp(row[asset] - price_short)
            data.loc[idx,'nav_short'] = pos * ret_short * np.exp(price_short) 
            data.loc[idx,'NAV'] += data.loc[idx,'nav_short'] + data.loc[idx,'nav_long'] 
            nav = data.loc[idx,'NAV']
            if row.signal==0:
                # Trade close.
                close = True
                data.loc[idx:,'NAV'] = nav
        else:
            # Trade open.
            if row.signal!=0:
                close = False 
                asset = asset2 if row.signal==1 else asset1
                price_short = row[asset]
                asset = asset1 if row.signal==1 else asset2
                price_long = row[asset]
                pos = round(nav_long*row.hedge_ratio/price_long,2)
                data.loc[idx,'pos'] = pos

        flag=row.signal
        asset = asset1 if row.signal==1 else asset2
        price_long = row[asset]

    data['port_ret'] = data['NAV']/data['NAV'].shift(1)
#    data.replace([np.inf,-np.inf,0], 1, inplace=True)
    plt.plot(np.cumprod(data['port_ret']), color='navy')
    pdb.set_trace()
    return 
