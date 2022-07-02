
import pdb
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



def interpolate_spline(data, col_list):

    
    intp_data = pd.DataFrame()
    col = col_list[0]
#    intp_data['air_temp_51101h'] = data.loc[(data.index>='2014-01-01') & \
#                                (data.index<'2015-01-01')]['air_temp_51101h']
    intp_data[col] = data[[col]]
    intp_data['xaxis'] = range(data.shape[0])
    intp_data = intp_data.loc[intp_data[col].notnull()]
    cs = CubicSpline(intp_data.xaxis, intp_data[col])
    xs = list(set(range(data.shape[0])) - set(intp_data.xaxis))
    ys = cs(xs)

    date_null = data[data[col].isnull()].index
    data.loc[date_null,col] = ys

    data[col + '_was_null'] = 0
    data.at[date_null,col + '_was_null'] = 1

    fig, ax = plt.subplots()
    plt.style.use('seaborn')
    ax.plot(data.index, data[col],label=col)
    ax.scatter(date_null, data.loc[date_null][col],label=col,marker='o')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(which='major',linewidth=0.5)
    ax.set_ylabel('Temp.')
    ax.set_xlabel('Date')
    ax.set_title('Station: 51101h; Air Temperature')
    pdb.set_trace()


    return 



