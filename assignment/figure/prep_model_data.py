
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

import sys
import pdb

def prep_model_data(df, no_encode=False):

    # Drop: First payment and Maturity date
    df.drop(['first_pmt_date','mat_date'],axis=1,inplace=True)
    
    # Drop variables because of zero variace. 
    df.drop(['prog_ind','prp_val_method', 'io_ind', \
            'amtr_type'],axis=1, inplace=True)

    # Columns that have `object` data types (`dtypes`)
    cols = df.columns[df.dtypes==object]
    cols = cols.append(pd.Index(['post_code','msa']))

    # Convert `pre_harp` to 'N' and 'Y' 
    df.loc[df['pre_harp'] != 'N','pre_harp'] = 'Y'

    # Drop loan sequence number
    cols = cols.drop(['loan_seq'])

    # Enumerate the dependent variable.
    df['def_bin'] = np.zeros(len(df))
    df.loc[(df.zero_def_bin==1) | (df.del_def_bin==1),'def_bin'] = 1.0
    df.set_index('loan_seq',inplace=True,drop=True)

    # For CatBoost algo. and targer encoding, we don't perform coding here.
    if no_encode:
#        cols = df.columns[df.dtypes == 'object']
        df[cols] = df[cols].astype('category')
        df.to_csv('final_data_cat.csv')
        return df

    # Columns that are ordinal. These can be encoded into `Labels`
    col_label = ['prp_state','seller_name','serv_name','pre_harp']
    # ********** To disable one-hot-code set col_label = col ********
    col_label = cols

    for col in col_label:
        df[col] = df[col].astype('category').cat.codes

    # Columns that are non-ordinal: these will be one-hot-encoded
    col_one_hot = set(cols) - set(col_label)

    #creating instance of one-hot-encoder
    encoder = OneHotEncoder(handle_unknown='ignore')

    for col in col_one_hot:
        # Perform one-hot encoding on `col` column 
        encd_df = pd.DataFrame(encoder.fit_transform(df[[col]]).toarray())
        # Rename the column to `col_n`
        encd_df = encd_df.add_prefix(col + '_')
        # Join the new variable to the dataframe
        df.reset_index(inplace=True,drop=True)
        df = df.join(encd_df)
        df.drop(col,axis=1,inplace=True)

    # df.to_csv('final_data.csv')
    return df 


def target_encoder(X,y):
    cat_features = X.select_dtypes(include='category').columns
    target_enc = ce.TargetEncoder(cols=cat_features)
    # Transform the features, rename the cols with _target suffix, and join df
    target_enc.fit(X[cat_features], y)
    X = X.join(target_enc.transform(X[cat_features]).add_suffix('_encde'))
    X.drop(cat_features,axis=1,inplace=True)
    return X 


