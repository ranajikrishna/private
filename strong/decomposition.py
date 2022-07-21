
import sys
import pdb
import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose


def data_decompose(data, model='additive'):

    print(model)
    col = 'wave_height_51201h'
#    res = seasonal_decompose(data[col], model=model)
    res = STL(data[col]).fit()
#    plt = res.plot()

    return res 
