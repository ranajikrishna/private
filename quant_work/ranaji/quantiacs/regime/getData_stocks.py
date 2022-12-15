
import pdb
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import xarray as xr

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata


def getData_stocks(start_date,end_date):

    data = qndata.stocks_load_data(min_date=start_date,max_date=end_date, dims=("field", "time", "asset"))
    data_df = pd.DataFrame(columns=data.sel(field=['open'])['asset'].values,\
                            index=data.sel(field=['open'])['time'].data.ravel())
    for ast in data_df.columns.values:
        data_df[ast] = data.sel(asset=ast,field='open').data.ravel()

    data_df.dropna(axis=1,inplace=True)
    return data_df 

