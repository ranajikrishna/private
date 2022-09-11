
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
#    pair_df['ratio'] = pair_df[pair[0]]/pair_df[pair[1]]

    win_reg = 180
    exog = sm.add_constant(pair_df[pair[1]])
    endog = pair_df[pair[0]]
    rols = RollingOLS(endog, exog, window=win_reg)
    rres = rols.fit()
    params = rres.params.copy()
    params.rename(columns={pair[1]:'hedge_ratio', "const":'spread'},inplace=True)
    pair_df = pair_df.join(params)
    
    win_zsc = win_reg
   # pair_df['zscore'] = (pair_df['ratio']-pair_df['ratio'].rolling(win_zsc).mean())/ \
   #                                pair_df['ratio'].rolling(win_zsc).std()

    pair_df['zscore'] = (pair_df['spread']-pair_df['spread'].rolling(win_zsc).mean())/ \
                                    pair_df['spread'].rolling(win_zsc).std()
#    zscore = []
#    for itr in range(win,pair_df.shape[0]):
#        zscore.append((pair_df.iloc[itr]['ratio'] - pair_df.iloc[0:itr]['ratio'].mean())/\
#                pair_df.iloc[0:itr]['ratio'].std())
#    zscore = [np.nan]*win + zscore
#    pair_df['zscore'] = zscore
    pair_df['signal'] = 0

    up,dw = 2,-2
    stop_loss = 3
    pair_df.loc[pair_df.zscore>=up,'signal'] = -1
    pair_df.loc[pair_df.zscore<=dw,'signal'] = 1

    flag = 0
    for idx,row in pair_df.iterrows():
#        if idx==pd.Timestamp('2016-10-26'):
#            pdb.set_trace()
        if (flag == 0 and pair_df.loc[idx]['signal'] ==0):
            continue;
        elif (flag != 0):
            if (abs(row.zscore)<0.5 or abs(row.zscore)>=stop_loss):
                flag = 0
            else:
                pair_df.loc[idx,'signal'] = flag
        else:
            flag += pair_df.loc[idx]['signal']
            
    pair_df.loc[abs(pair_df.zscore)>=stop_loss,'signal'] = 0
    pair_df.loc[pair_df['hedge_ratio']<0.1,'signal'] = 0
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
