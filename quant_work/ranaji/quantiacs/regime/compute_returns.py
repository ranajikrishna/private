
import pdb
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

def compute_returns(data):
    '''
    We use hedge account to compute returns of the pairs trading trategy. 
    NOTE:
    [1] Refer to `hedge_accounting.example_final` on goggle sheet for an example.
    The returns computed are similar to those computed by the Quantiacs fxn. 
    `qnt.output`. There are descrepancies between the two because the latter 
    accounts for slippage and transaction costs. 
    [2] These returns are computed for CLOSE prices. OPEN prices would have a 
    different logic.
    '''
    asset1, asset2 = data.columns[0], data.columns[1]
    data['nav_long'], data['nav_short'], data['NAV'], data['endog_wgt'], data['exog_wgt'],data['port_ret'] = 0,0,1,0,0,0
    nav_long,nav,flag = 1,1,0
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
            ret_short = 2 - np.exp(row[asset] - price_short)
            data.loc[idx,'nav_short'] = ret_short * nav_short
            data.loc[idx,'NAV'] = data.loc[idx,'nav_short'] + data.loc[idx,'nav_long'] 
            # Net Asset Value (NAV) of the portfolio.
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
                ratio = row.hedge_ratio if row.signal==1 else 1/row.hedge_ratio
                data.loc[idx,'endog_wgt'] = 1/(1 + row.hedge_ratio)
                data.loc[idx,'exog_wgt'] = row.hedge_ratio/(1 + row.hedge_ratio)
                long_wgt = data.loc[idx,'endog_wgt'] if row.signal==1 else data.loc[idx,'exog_wgt']
                short_wgt = data.loc[idx,'exog_wgt'] if row.signal==1 else data.loc[idx,'endog_wgt']
                nav_long = long_wgt * nav
                nav_short = short_wgt * nav

        flag=row.signal
        asset = asset1 if row.signal==1 else asset2
        price_long = row[asset]

    data['port_ret'] = data['NAV']/data['NAV'].shift(1)
#    plt.plot(np.cumprod(data['port_ret']), color='navy')
    return data 
