
# In your final submission you can remove/deactivate all the other cells to reduce the checking time.
# The checking system will run this book multiple times for every trading day within the in-sample period.
# Every pass the available data will be isolated till the current day.
# qnt.backtester is optimized to work with the checking system.
# The checking system will override test_period=1 to make your strategy to produce weights for 1 day per pass.

import os
os.environ['API_KEY'] = "{f4ef60fd-ae0b-4865-9d94-7cc9000dba05}"
import xarray as xr
import numpy as np
import pandas as pd

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata
import qnt.xr_talib as xrtl
import xarray.ufuncs as xruf
import qnt.ta as qnta

import pdb


def load_data(period):
    return qndata.futures_load_data(tail=period)


def calc_positions(futures, ma_periods, roc_periods, sideways_threshold):
    """ Calculates positions for given data(futures) and parameters """
    close = futures.sel(field='close')
    
    # calculate MA 
    ma = qnta.lwma(close, ma_periods)
    # calcuate ROC
    roc = qnta.roc(ma, roc_periods)

    # positive trend direction
    positive_trend = roc > sideways_threshold
    # negtive trend direction
    negative_trend = roc < -sideways_threshold 
    # sideways
    sideways_trend = abs(roc) <= sideways_threshold
    
    # We suppose that a sideways trend after a positive trend is also positive
    side_positive_trend = positive_trend.where(sideways_trend == False).ffill('time').fillna(False)
    # and a sideways trend after a negative trend is also negative
    side_negative_trend = negative_trend.where(sideways_trend == False).ffill('time').fillna(False)

    # define signals
    buy_signal = positive_trend
    buy_stop_signal = side_negative_trend

    sell_signal = negative_trend
    sell_stop_signal = side_positive_trend

    # calc positions 
    position = close.copy(True)
    position[:] = np.nan
    position = xr.where(buy_signal, 1, position)
    position = xr.where(sell_signal, -1, position)
    position = xr.where(xruf.logical_and(buy_stop_signal, position.ffill('time') > 0), 0, position)
    position = xr.where(xruf.logical_and(sell_stop_signal, position.ffill('time') < 0), 0, position)
    position = position.ffill('time').fillna(0)

    return position


def calc_output_all(data, params):
    positions = data.sel(field='close').copy(True)
    positions[:] = np.nan
    for futures_name in params.keys(): 
        p = params[futures_name]
        futures_data = data.sel(asset=futures_name).dropna('time','any')
        p = calc_positions(futures_data, p['ma_periods'], p['roc_periods'], p['sideways_threshold'])
        positions.loc[{'asset':futures_name, 'time':p.time}] = p
    
    return positions

# say we select futures and their parameters for technical algorithm
params = {
    'F_NY': {
        'ma_periods': 200, 
        'roc_periods': 5, 
        'sideways_threshold': 2,
    },
    'F_GX': {
        'ma_periods': 200, 
        'roc_periods': 20, 
        'sideways_threshold': 2
    },
    'F_DX': {
        'ma_periods': 40, 
        'roc_periods': 6, 
        'sideways_threshold': 1
    },
}
futures_list = list(params.keys())


def strategy(data):
    output = calc_output_all(data.sel(asset = futures_list), params)
    tmp = output.to_pandas()
    tmp = tmp.rename_axis('asset',axis="columns").rename_axis('time')
    output1 = tmp.to_xarray()
    pdb.set_trace()
    coordinates={'time':(['time'],[output.time.values]),'asset':(['asset',output.asset.values]),'field':'close'}
    op1=xr.DataArray(data=output.values,coords=coords,dims=['time','asset'])
    return output.isel(time=-1)

weights = qnbt.backtest(
    competition_type="futures",
    load_data=load_data,
    lookback_period=2*365,
    start_date='2018-01-01',
    strategy=strategy
)

def main():
    # create data
    data = np.random.randn(2, 3, 3)

    # create coordinates
    rows = [1,2]
    cols = [1,2,3]
    row_meshgrid, col_meshgrid = np.meshgrid(rows, cols, indexing='ij')
    time = pd.date_range("2000-01-01", periods=3)

    # put data into a dataset
    ds = xr.Dataset(
        data_vars=dict(
            variable=(["x","y","time"], data, {"units":"m/s"})
        ),
        coords=dict(
            row=(["x","y"], row_meshgrid),
            col=(["x","y"], col_meshgrid),
            time=(["time"], time)
        ),
        attrs=dict(description="coords with matrices"),
    )
    return 

#main()

