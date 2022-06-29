
import os
api_key = 'f4ef60fd-ae0b-4865-9d94-7cc9000dba05'
os.environ['API_KEY'] = api_key
import xarray as xr

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
import matplotlib.cbook as cbook 
import multiprocessing as mp

import pdb

def plot(price, asset):
    #asset = set(asset) - set(['DOGE','ETH'])
    fig, ax = plt.subplots()
    price[asset].plot()
    return 

def correlated_asset(corr, ast):
    '''
    Returns all intercorrelated assests. This means that all the asset pairs 
    are correlated to each other.
    '''
    row = corr[corr[ast]==1].index
    corr_ast = corr.loc[row][row]
    corr_sum = corr_ast.sum().sort_values()
    while (len(corr_sum) != corr_ast.sum().sort_values()[0]): 
        drop_ast = corr_sum.index[0]
        corr_ast.drop(index=drop_ast,columns=drop_ast,inplace=True)
        corr_sum = corr_ast.sum().sort_values()
    return corr_ast.index.values.tolist()


def parallel_fxn(arg):
    return arg[0].rolling(arg[1]).corr(arg[2]).median()


def window_corr(price, win_lb, win_la):

    corr_win = pd.DataFrame(columns=price.columns, index=price.columns)
    lhs = price
    rhs = price.shift(win_la).dropna()
    assets = price.columns.to_list()
    pool = mp.Pool(4)
    i = 0
    for x in price.columns:
        args = [(lhs[x],win_lb,rhs[i]) for i in assets]
        corr = pool.map(parallel_fxn, args)
        corr_win[x][i:] = corr
        i+=1
        assets.remove(x)
        print(i)

    pdb.set_trace()
#    for x in price.columns:
#        for y in assets:
#            corr=lhs[x].rolling(win_lb).corr(rhs[y])
#            corr_win[x][y] = corr.median()
#        assets.remove(x)
#
    
    return 

def correlation(price, lag):
    '''
    Compute lagged-correlations
    '''
    if lag==0:
        return price.corr()
    lhs = price.shift(lag).dropna()
    rhs = price[0:-lag]
    numerator = np.dot((lhs-np.mean(lhs)).transpose(),(rhs-np.mean(rhs)))
    denominator = np.dot(np.array([lhs.std()]).transpose(),np.array([rhs.std()]))

    xcorr = pd.DataFrame((1/len(rhs))*numerator/denominator,columns=lhs.columns, \
                                                            index=lhs.columns)
    return xcorr 

def load_data(var):
    #universe = qndata.cryptodaily_load_data(min_date=var)
    universe = qndata.stocks_load_data(min_date=var)
    open_ = universe.sel(field="open").to_pandas()
    low = universe.sel(field="low").to_pandas()
    high  = universe.sel(field="high").to_pandas()
    close = universe.sel(field="close").to_pandas()
    return open_,low,high,close

def main():
    min_date="2018-01-01"
    open_,low,high,close = load_data(min_date)
    close.dropna(axis=1,inplace=True)
    lag = 10
    corr = correlation(close,lag)
    
    val = 0.8
    corr_bit = (abs(corr) >= val).astype(int)
    #window_corr(close, 10, 5)
    #corr_asset = correlated_asset(corr_bit, 'ADA')
    #corr_asset = correlated_asset(corr_bit, 'AMEX:CIX')
    #corr_asset = correlated_asset(corr_bit, 'NYSE:KEG')
    #corr_win['NYSE:HCA']['NYSE:KEG']
    #corr_win['NYSE:KEG']['NYSE:MKL']
    #corr_win['NYSE:KEG']['NYSE:PVH']
    corr_asset = ['NYSE:HCA','NYSE:KEG']
    print (corr.loc[corr_asset][corr_asset])
    plot(close,corr_asset)
    pdb.set_trace()
    return 

if __name__ == '__main__':

    status = main()

