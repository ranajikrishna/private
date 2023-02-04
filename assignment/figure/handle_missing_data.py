
import pandas as pd
import numpy as np

import sys
import pdb



def handle_missing_data(df):
    '''
    Function to examine columns with empty strings.
    '''
    is_emp = ([])
    for col in df.columns:
        if df[col].isna:
            is_emp.append(col)

    # Handle missing MSA values.
    df.msa.fillna(np.max(df.msa)+1,inplace=True)

    # Handle super conforming flag
    df.super.fillna('N',inplace=True)

    # Handle pre-HARP loan
    # Fill empty data with N (Not a pre-HARP)
    df.pre_harp.fillna('N',inplace=True)
    # Variable to identify: N-Not a pre-HARP, A-ARM, F-FRM
    df['pre_harp_status'] = df.pre_harp.str[0]

    # Handle HARP indicator
    df.hard_ind.fillna('N', inplace=True)

    # Drop loans that do not have monthly performance data
    df = df[df['del_def_bin'].notna()]

    # Replace NaN in zero balance removal upb by 0.
    df.zero_bal_upb.fillna(0,inplace=True)

    return df 
