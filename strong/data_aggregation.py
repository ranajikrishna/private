
import sys
import pdb
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def event_pop(data):

    col = 'wave_height_51201h'
    cnt = sum(data[col]>=3)
    data['surf_count'] = cnt
    return data

def event_count(data):
    return sum(data>3)

def data_agg(data, col):

    #W-MON
    data['date'] = pd.to_datetime(data.index) 
#    data_week = data.groupby([pd.Grouper(key='date',freq='W-MON')])[col].mean()
#    data_week['surf_day'] = data.groupby([pd.Grouper(key='date',freq='W-MON')])[col].agg([event_count])

    data_week = data[col].rolling(window=7).mean()
    data_week['surf_day'] = data[col].rolling(window=7).agg([event_count])

#    data_week = data.groupby([pd.Grouper(key='date',freq='W-MON')]).apply(event_pop)
    return data_week.fillna(0)
