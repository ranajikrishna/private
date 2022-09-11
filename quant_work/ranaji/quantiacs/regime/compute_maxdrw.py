
import pdb
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

def compute_maxdrw(data):
    max_ret,min_ret = 0,0
    data['max_drw'] = 0
    for idx,row in data.iterrows():
        max_ret,min_ret = max(row['MTM'],max_ret),min(row['MTM'],min_ret)
        data.loc[idx,'max_drw'] = min_ret - max_ret 
    return data 

