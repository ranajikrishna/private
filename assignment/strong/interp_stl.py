
import pdb
import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.seasonal import STL


def interpolate_stl(data, col_list):

    intp_data = pd.DataFrame()
    col = col_list[0]

    intp_data[col] = pd.concat([data.loc[(data.index<'2014-01-01')][col], \
                            data.loc[(data.index>='2015-01-01')][col]])
    pdb.set_trace()
    res = STL(intp_data[col]).fit()

    return 
