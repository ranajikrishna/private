
import pdb
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


def compute_returns(data):
    asset1, asset2 = data.columns[0], data.columns[1]
    data['nav_long'], data['nav_short'], data['NAV'], data['ret_port'] = 0,0,0,0
    nav_long,nav,flag = 1,0,0
    close = True
    for idx,row in data.iterrows():
        if idx==pd.Timestamp('2020-12-18'):
            pdb.set_trace()
        data.loc[idx,'nav_long'] += nav_long
        if (row.signal+flag!=0 and not close):
            asset = asset1 if flag==1 else asset2
            ret_long = row[asset] - price_long 
            nav_long = np.exp(ret_long) * nav_long
            data.loc[idx,'nav_long'] = nav_long 
            asset=asset2 if flag==1 else asset1
            ret_short = 1 - np.exp(row[asset] - price_short)
            data.loc[idx,'nav_short'] = ret_short * np.exp(price_short) 
            data.loc[idx,'NAV'] += data.loc[idx,'nav_short'] + data.loc[idx,'nav_long'] 
            nav = data.loc[idx,'NAV']
            if row.signal==0:
                close = True
                data.loc[idx:,'NAV'] = nav
        else:
            if row.signal!=0:
                close = False 
                asset = asset2 if row.signal==1 else asset1
                price_short = row[asset]

        flag=row.signal
        asset = asset1 if row.signal==1 else asset2
        price_long = row[asset]

    pdb.set_trace()
    return 



def compute_return3(data):
    asset1, asset2 = data.columns[0], data.columns[1]
    data['nav_long'], data['nav_short'], data['NAV'], data['ret_port'] = 0,0,0,0
    close = True
    flag = 0
    nav_long,nav = 1,0
    for idx,row in data.iterrows():
        data.loc[idx,'nav_long'] += nav_long
        data.loc[idx,'NAV'] += nav  ##+ np.exp(ret_short) - 1) 
        if (row.signal+flag)!=0:
            if close:
                flag=row.signal
                asset = asset2 if flag==1 else asset1
                price_short = row[asset]
                close = False
            else:   
                asset = asset1 if flag==1 else asset2
                ret_long = row[asset] - price_long 
                nav_long = np.exp(ret_long) * nav_long
                data.loc[idx,'nav_long'] = nav_long # * (data.iloc[idx-1,'MTM_long']) 
                asset=asset2 if flag==1 else asset1
                ret_short = 1 - np.exp(row[asset] - price_short)
                data.loc[idx,'nav_short'] = ret_short * np.exp(price_short) # (da##t.ailoc[idx-1,'MTM_short']) 
                data.loc[idx,'NAV'] = data.loc[idx,'nav_short'] + data.loc[idx,'nav_long'] + nav  ##+ np.exp(ret_short) - 1) 

            if row.signal==0:
                flag=row.signal
                nav = data.loc[idx,'NAV']
                close = True
    return 



def compute_returns1(data):
    asset1, asset2 = data.columns[0], data.columns[1]
    data['nav_long'], data['nav_short'], data['NAV'], data['ret_port'] = 0,0,0,0
    close = False
    flag = 0
    nav_long,nav = 1,0
    for idx,row in data.iterrows():
        if row.signal!=0 or close:
            flag=row.signal
            if not close:
#                asset = asset1 if flag==1 else asset2
#                price_long = row[asset]
                data.loc[idx,'nav_long'] += nav_long
                data.loc[idx,'NAV'] = nav  ##+ np.exp(ret_short) - 1) 
                asset = asset2 if flag==1 else asset1
                price_short = row[asset]
                close = True
            else:
                # === Mark-to-market ===
                asset = asset1 if flag==1 else asset2
                ret_long = row[asset] - price_long 
                nav_long = np.exp(ret_long) * nav_long
                data.loc[idx,'nav_long'] = nav_long # * (data.iloc[idx-1,'MTM_long']) 
                asset=asset2 if flag==1 else asset1
                ret_short = 1 - np.exp(row[asset] - price_short)
                data.loc[idx,'nav_short'] = ret_short * np.exp(price_short) # (da##t.ailoc[idx-1,'MTM_short']) 
                data.loc[idx,'NAV'] = data.loc[idx,'nav_short'] + data.loc[idx,'nav_long'] + nav  ##+ np.exp(ret_short) - 1) 
                #nav = data.loc[idx,'NAV']
        else:
            data.loc[idx,'nav_long'] += nav_long
            data.loc[idx,'NAV'] = nav  ##+ np.exp(ret_short) - 1) 
            close = False


        asset = asset1 if flag==1 else asset2
        price_long = row[asset]
#        asset = asset2 if flag==1 else asset1
#        price_short = row[asset]

    data['ret_port']= data['NAV']/data['NAV'].shift(1) -1 
    pdb.set_trace()
#    plt.plot(np.exp(np.cumsum(data.ret_port),color='navy'))
    return data

def compute_returns2(data):
    asset1, asset2 = data.columns[0], data.columns[1]
    data['MTM_long'], data['MTM_short'] = 0,0
    ret = defaultdict(list)
    flag = 0
    close = False
    for idx,row in data.iterrows():
        if row.signal!=flag:
            flag=row.signal
            if not close:
                price_open = row
                close = True
            else:
                ret[asset1].append(flag_prev*(row[asset1]-price_open[asset1]))
                ret[asset2].append(-1*flag_prev*(row[asset2]-price_open[asset2]))
                close = False
        flag_prev = flag
    pdb.set_trace()
    return  
