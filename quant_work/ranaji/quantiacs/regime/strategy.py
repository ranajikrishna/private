
import sys
import pdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import matplotlib.pyplot as plt

def strat_one(data_df,pair):
    pair_df = pd.DataFrame({pair[0]:data_df[pair[0]], \
                                    pair[1]:data_df[pair[1]]})
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
    pair_df['signal'] = 0

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
        else:
            # Start trade: update flag.
            flag += row.signal
            
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

    return pair_df 
