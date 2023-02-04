
import pdb
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import interp_seasonal as intrp_sea
import statsmodels.tsa.stattools as smt


def examine_correl(wide):


    col_list = ['air_temp_51000h', 'air_temp_51101h', 'average_wave_period_51000h',
       'average_wave_period_51101h', 'average_wave_period_51201h',
       'dominant_wave_period_51000h', 'dominant_wave_period_51101h',
       'dominant_wave_period_51201h', 'wave_height_51000h', 'wave_height_51101h',
       'wave_height_51201h']

    data = pd.DataFrame(columns=col_list)

    for col in col_list: 
        data[col] = intrp_sea.interpolate_seasonal(wide,[col]).drop(columns='yr_mean')
        
    xcorr = pd.DataFrame(columns=col_list)
    wb = ['wave_height_51201h']
    for col in col_list:
        xcorr[col] = smt.ccf(data[wb],data[col])

    pdb.set_trace()

    return 
