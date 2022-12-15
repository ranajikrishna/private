
import sys
import pdb
import numpy as np
import pandas as pd
import pickle as pk

import matplotlib as plt
import seaborn as sns
import statsmodels
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts


def cointegrated_pairs(data,load=True):
    if not(load):
        p_values = pd.DataFrame(columns=data.columns,index=data.columns)
        skip = 0
        for asset in data.columns.values:
            skip += 1
            for pair in data.columns.values[skip:]:
                p_values.loc[asset,pair] = coint(data[asset],data[pair])[1]

        p_values.to_pickle("./p_values_log.pk")

    pdb.set_trace()
    file = open("./p_values_log.pk",'rb')
    p_values = pk.load(file)
    file.close()
    bool_pairs = p_values<0.001
    coint_pairs = bool_pairs.reset_index().melt(id_vars='index').query('value == True')
    coint_pairs = coint_pairs[coint_pairs['index']!=coint_pairs['variable']]
    coint_pairs.rename(columns={'index':'asset1','variable':'asset2'},inplace=True)

    for idx,row in coint_pairs.iterrows():
        coint_pairs.loc[idx,'value'] = p_values.loc[coint_pairs.loc[idx]['asset1'],coint_pairs.loc[idx]['asset2']]
    
    return coint_pairs.sort_values(by='value')
