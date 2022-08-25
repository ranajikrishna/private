
import pdb
import sys
import pandas as pd
import matplotlib as plt
import numpy as np

import xarray as xr

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata

#from statsmodels import regression



def pctChange(price,freq):
    return [price[i+freq-1]/price[i] -1 for i in range(0,len(price)-freq+1)]


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):

    end_date = 20201230
    start_date = 20141230
    freq = 30 
    index = [(DATE>=start_date) & (DATE<=end_date)]

#    start_date = 20140730
#    end_date = 20150730
#    offset_index = [(DATE>=start_date) & (DATE<=end_date)]

    open_price = OPEN[index]
    high_price = HIGH[index]
    low_price = LOW[index]
    close_price = CLOSE[index]
    vol_price = VOL[index]
    exposure_price = exposure[index]
    equity_price = equity[index]

    pdb.set_trace()
    asset1 = pctChange(close_price[:,0],freq)
    asset2 = pctChange(close_price[:,1],freq)
    bench = pctChange(close_price[:,2],freq)
    treasury_ret = close_price[:,3]

    pdb.set_trace()
    df = pd.DataFrame({'R1':asset1, 'R2':asset2, 'SPY': bench, \
                       'const': np.ones(len(asset1))})
    
    pdb.set_trace()
    ols_1 = regression.linear_model.OLS(df['R1'], df[['SPY', 'const']])
    ols_1_fit = ols_1.fit()
    ols_2 = regression.linear_model.OLS(df['R2'], df[['SPY', 'const']])
    ols_2_fit = ols_2.fit()


    return weights, settings



def mySettings():
    ''' Define your trading system settings here '''

    settings = {}

    # S&P 100 stocks
    settings['markets']=['GOOGL','MSFT','F_ES','F_TY']
#    settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
#    'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
#    'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
#    'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
#    'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
#    'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
#    'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
#    'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
#    'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts
#    settings['markets'] = ['CASH', 'F_AD', 'F_AE']
#                           'F_AH', 'F_AX', 'F_BC', 'F_BG', 'F_BO', 'F_BP', 'F_C',  'F_CA',
#                           'F_CC', 'F_CD', 'F_CF', 'F_CL', 'F_CT', 'F_DL', 'F_DM', 'F_DT', 'F_DX', 'F_DZ', 'F_EB',
#                           'F_EC', 'F_ED', 'F_ES', 'F_F',  'F_FB', 'F_FC', 'F_FL', 'F_FM', 'F_FP', 'F_FV', 'F_FY',
#                           'F_GC', 'F_GD', 'F_GS', 'F_GX', 'F_HG', 'F_HO', 'F_HP', 'F_JY', 'F_KC', 'F_LB', 'F_LC',
#                           'F_LN', 'F_LQ', 'F_LR', 'F_LU', 'F_LX', 'F_MD', 'F_MP', 'F_ND', 'F_NG', 'F_NQ', 'F_NR',
#                           'F_NY', 'F_O',  'F_OJ', 'F_PA', 'F_PL', 'F_PQ', 'F_RB', 'F_RF', 'F_RP', 'F_RR', 'F_RU',
#                           'F_RY', 'F_S',  'F_SB', 'F_SF', 'F_SH', 'F_SI', 'F_SM', 'F_SS', 'F_SX', 'F_TR', 'F_TU',
#                           'F_TY', 'F_UB', 'F_US', 'F_UZ', 'F_VF', 'F_VT', 'F_VW', 'F_VX',  'F_W', 'F_XX', 'F_YM',
#                           'F_ZQ']

    settings['lookback'] = -1
    settings['budget'] = 10**6
    settings['slippage'] = 0.05

    return settings


def main():
    pdb.set_trace()
    sett = mySettings()

    return 


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    status = main()
    sys.exit()
    
    #from quantiacsToolbox.quantiacsToolbox import runts, optimize

    results = runts(__file__)
    #optimize(__file__)




