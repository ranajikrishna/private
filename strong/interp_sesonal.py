
import pdb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def interpolate_seasonal(data, col_list):

    col = col_list[0]
    intp_data = pd.DataFrame({col:data[col]}, columns = [col,'roll_mean'])
    win = 120

    pdb.set_trace()

    return 
