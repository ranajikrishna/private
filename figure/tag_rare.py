
import sys
import numpy as np
import pandas as pd
import random

import pdb


def create_rare_tag(df):
    
    N = 100
    col_tag = ['post_code','msa']

    for col in col_tag:
        ind = random.sample(df.index.to_list(),N)
        df[col][ind]


    pdb.set_trace()

    return 
