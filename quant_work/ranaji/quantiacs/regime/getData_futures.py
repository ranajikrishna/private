
import pdb
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata

def getData_futures(asset_tkr,start_date,end_date):

    data = qndata.futures.load_data(assets=[asset_tkr], min_date=start_date, \
                        max_date=end_date, dims=("field", "time", "asset"))

    data_df = pd.DataFrame({'open':data.sel(field='open').data.ravel(),
                  'close': data.sel(field='close').data.ravel(),
                  'high': data.sel(field='high').data.ravel(),
                  'low': data.sel(field='low').data.ravel(),
                  'vol_day': data.sel(field='vol').data.ravel(),
                  'open_int': data.sel(field='oi').data.ravel(),
                  'ctr_roll_ovr': data.sel(field='roll').data.ravel()
                  }, index=data.time.data)
    
    return data_df 

