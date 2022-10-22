
import xarray as xr

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata
import qnt.output as qnout


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import matplotlib.pyplot as plt
import compute_returns as cr
import pdb

def load_data(period):
    return qndata.stocks_load_data(tail=period, assets=["NAS:BKNG", "NAS:CSCO"])

def strategy(data):
    pair = ('NAS:BKNG','NAS:CSCO')
    data_df = pd.DataFrame({'NAS:BKNG':data.sel(asset='NAS:BKNG',field='open').data,\
            'NAS:CSCO':data.sel(asset='NAS:CSCO',field='open').data.ravel()},\
            index=data.sel(field=['open'])['time'].data)

    pair_df = np.log(data_df)
    win_reg = 180
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
    pair_df['signal'], pair_df['NAS:CSCO_wgt'], pair_df['NAS:BKNG_wgt'] = 0,0,0

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
            continue;
        elif (flag != 0):
            # Trade has started: check if it needs to continue.
            if (abs(row.zscore)<limit_zscore_dw or abs(row.zscore)>=limit_zscore_up or row.hedge_ratio<=limit_hedge):
                # End trade due to end conditions.
                flag = 0
            else:
                # Continue trade: populate trade signal.
                pair_df.loc[idx,'signal'] = flag
                pair_df.loc[idx,'NAS:BKNG_wgt'] = wgt 
                pair_df.loc[idx,'NAS:CSCO_wgt'] = 1 - wgt 
        else:
            # Start trade: update flag.
            flag += row.signal
            wgt = 1/(1 + row['hedge_ratio']) if flag==1 else row['hedge_ratio']/(1 + row['hedge_ratio'])
            pair_df.loc[idx,'NAS:BKNG_wgt'] = wgt 
            pair_df.loc[idx,'NAS:CSCO_wgt'] = 1 - wgt 

    weight = pd.DataFrame({'NAS:BKNG': pair_df['NAS:BKNG_wgt'],\
                           'NAS:CSCO':pair_df['NAS:CSCO_wgt']})
    if False:
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
    
    weight = weight.rename_axis('asset',axis="columns").rename_axis('time')
    #return weight.isel(time=-1)
    return weight.to_xarray()

period = 3650 
data = load_data(period)
weights = strategy(data)
pdb.set_trace()
qnout.write(weights)
 

#weights = qnbt.backtest(
#    competition_type="stocks",
#    load_data=load_data,
#    lookback_period=5*365,
#    start_date='2015-01-01',
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
