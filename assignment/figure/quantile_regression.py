
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV

import sys
import six
sys.modules['sklearn.externals.six'] = six
import mlrose

from skgarden import RandomForestQuantileRegressor
from scipy.stats import randint

import pdb

def rfq_reg(df):
    df = df.loc[df.def_bin==1]
    X, y = df.drop(['zero_def_bin','del_def_bin','def_bin', 'def_time'],\
            axis=1),df['def_time']
    para_grid = {'n_estimators': [50, 100, 150],
                 'max_depth': np.linspace(1,30,30), \
                 'max_leaf_nodes': np.linspace(2,100,10).astype(int)}
    rfq_model = RandomForestQuantileRegressor()

    hyp_reg = RandomizedSearchCV(rfq_model, param_distributions = para_grid,cv=3,\
                                                        n_jobs=None, verbose=2)

    hyp_reg.fit(X,y)
    hyp_reg.best_estimator_
        
    pdb.set_trace()
    return 


def rfq_reg_cv(df):
    df = df.loc[df.def_bin==1]
    X, y = df.drop(['zero_def_bin','del_def_bin','def_bin', 'def_time'],\
            axis=1),df['def_time']
    folds = KFold(n_splits=5)
    rfq_model = RandomForestQuantileRegressor(n_estimators=50,max_leaf_nodes=10,\
            max_depth=6)
    chk=False
    pdb.set_trace()
    for i, (train_index, test_index) in enumerate(folds.split(X,y)):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        rfq_model.fit(X_train,y_train)

        if chk:
            tmp_up = pd.DataFrame(np.ceil(rfq_model.predict(X_test, quantile=95)),\
                                    index=y_test.index, columns=['upper'])
            pred_up = pred_up.append(tmp_up)

            tmp_dw = pd.DataFrame(np.ceil(rfq_model.predict(X_test, quantile=5)),\
                                    index=y_test.index, columns=['lower'])
            pred_dw = pred_dw.append(tmp_dw)

        else:
            pred_up = pd.DataFrame(np.ceil(rfq_model.predict(X_test, quantile=95)),\
                                    index=y_test.index,columns=['upper'])

            pred_dw = pd.DataFrame(np.ceil(rfq_model.predict(X_test, quantile=5)),\
                                    index=y_test.index,columns=['lower'])
            chk = True
        
    df=df.join(pred_up,how='left')        
    df=df.join(pred_dw,how='left')        
#    df.to_pickle("./rfq_reg.pk")
    pdb.set_trace()
    return

