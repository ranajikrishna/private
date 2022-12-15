
import xarray as xr

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata
import qnt.output as output
import qnt.stats as qnstats

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import matplotlib.pyplot as plt
import compute_returns as cr
import pdb

def load_data(period,pair):
    #return qndata.stocks_load_data(tail=period, assets=["NAS:BKNG", "NAS:CSCO"])
    #return qndata.stocks_load_data(tail=period, assets=["NAS:ADBE", "NAS:AMZN"])
    return qndata.stocks_load_data(tail=period, assets=[pair[0], pair[1]])

def strategy(data):
    pair = (data.asset.values[0], data.asset.values[1]) # Pair of assets in cointegrated order: pair = (Endogenous, Exogenous)
    data_df = pd.DataFrame({pair[0]:data.sel(asset=pair[0],field='open').data,\
            pair[1]:data.sel(asset=pair[1],field='open').data.ravel()},\
            index=data.sel(field=['open'])['time'].data)

    pair_df = np.log(data_df)
    win_reg = 60
    exog = sm.add_constant(pair_df[pair[1]])
    endog = pair_df[pair[0]]
    rols = RollingOLS(endog, exog, window=win_reg)
    rres = rols.fit()
    params = rres.params.copy()
    params.rename(columns={pair[1]:'hedge_ratio', "const":'spread'},inplace=True)
    pair_df = pair_df.join(params)
    
    win_zsc = win_reg
    pair_df['zscore'] = (pair_df['spread']-pair_df['spread'].rolling(win_zsc).mean())/ \
                                    pair_df['spread'].rolling(win_zsc).std()


    pair_wgt = (data.asset.values[0] + '_wgt', data.asset.values[1] + '_wgt') # Pair of assets in cointegrated order: pair = (Endogenous_wgt, Exogenous_wgt)
    pair_df['signal'], pair_df[pair_wgt[0]], pair_df[pair_wgt[1]] = 0,0,0

    # Trade-start conditions.
    up,dw = 2,-2
    pair_df.loc[((pair_df.zscore>=up) & (pair_df.hedge_ratio>=0.01)),'signal'] = -1 # Short spread
    pair_df.loc[((pair_df.zscore<=dw) & (pair_df.hedge_ratio>=0.01)),'signal'] = 1  # Long spread

    # Identify trade-end conditions.
    limit_zscore_up = 3
    limit_zscore_dw = 0.5
    limit_hedge = 0.1  
    pair_df.loc[abs(pair_df.zscore)>=limit_zscore_up,'signal'] = 0
    pair_df.loc[abs(pair_df.zscore)<=limit_zscore_dw,'signal'] = 0
    pair_df.loc[pair_df['hedge_ratio']<limit_hedge,'signal'] = 0

    # Populate trade signal. 
    flag = 0
    for idx,row in pair_df.iterrows():
#        if idx==pd.Timestamp('2013-12-16'):
#            pdb.set_trace()
        if (flag == 0 and row.signal==0):
            # Trade off
            continue;
        elif (flag != 0):
            # Trade has started: check if it needs to continue.
            if (abs(row.zscore)<limit_zscore_dw or abs(row.zscore)>=limit_zscore_up or row.hedge_ratio<=limit_hedge):
                # End trade due to end conditions.
                flag = 0
            else:
                # Continue trade: populate trade signal.
                pair_df.loc[idx,'signal'] = flag
                pair_df.loc[idx,pair_wgt[0]] = wgt 
                pair_df.loc[idx,pair_wgt[1]] = 1 - wgt 

        else:
            # Start trade: update flag.
            flag += row.signal
            wgt = 1/(1 + row['hedge_ratio'])
            pair_df.loc[idx,pair_wgt[0]] = wgt 
            pair_df.loc[idx,pair_wgt[1]] = 1 - wgt 

    pair_df[pair_wgt[0]] = pair_df.signal * pair_df[pair_wgt[0]] 
    pair_df[pair_wgt[1]] = -1*pair_df.signal * pair_df[pair_wgt[1]] 

    weight = pd.DataFrame({pair[0]: pair_df[pair_wgt[0]],\
                           pair[1]:pair_df[pair_wgt[1]]})

    if True:
        # === Plot === 
        fig, ax1 = plt.subplots() 
        ax1.plot(pair_df[pair[0]],label=pair[0]) 
        ax1.plot(pair_df[pair[1]],label=pair[1]) 
        ax1.legend()
        ax1.set_ylabel('Price')
  
        ax2 = ax1.twinx() 
        ax2.plot(pair_df['zscore'], color = 'blue',alpha=0.3,label=str(win_reg)+'-day ratio')  
        ax2.plot(pair_df['signal'],color='red')
        ax2.legend()
        ax2.set_ylabel('Ratio')
        ax1.grid()
        plt.xticks(rotation=45)
    
    ret_df = cr.compute_returns(pair_df)
    data = np.vstack((weight[pair[0]],weight[pair[1]])).T
    weight = xr.Dataset({"data":(["time","asset"],data)},\
        coords={"time":weight.index.values,"asset":[pair[0],pair[1]]})

    import pickle
    ret_df.to_pickle('./ret_df.pk')
    plt.plot(np.cumprod(ret_df['port_ret']), color='navy')
    pdb.set_trace()
    return weight.data

sel_pair = [('NAS:ADBE','NAS:AMZN'),('NAS:ADSK','NAS:AMZN'),('NAS:ADSK','NAS:ANSS'),
            ('NAS:BKNG','NAS:CHKP'),('NAS:WTW','NAS:XEL'),
            ('NAS:WTW','NYS:QGEN'), ('NAS:NTES','NYS:QGEN'),('NAS:BKNG','NAS:CSCO'),
            ('NAS:ANSS','NAS:WTW')]
#('NAS:APPL','NAS:ODFL'),

for pair in sel_pair:
    data = load_data(4000, pair)
    weights = strategy(data)
    output.write(weights)
    stat = qnstats.calc_stat(data, weights.sel(time=slice("2006-01-01", None)), slippage_factor=None,
                             roll_slippage_factor=None)
    performance = stat.to_pandas()["equity"]
    tmp = stat.to_pandas()
    ret_df = pd.read_pickle("./ret_df.pk")
    eqt_nav = pd.merge(tmp.equity,ret_df.NAV,how='left',left_index=True,right_index=True)
    pdb.set_trace()



#weights = qnbt.backtest(
#    competition_type="stocks",
#    load_data=load_data,
#    lookback_period=5*365,
#    start_date='2005-01-01',
#    strategy=strategy
#)

#def main():
#    period = 3650 
#    data = load_data(period)
#    tmp = strategy(data)
#    pdb.set_trace() 
#    return 
#
#main()
