
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import catboost as cb

import pdb
import sys

def boosted_reg_cv(df):
    pdb.set_trace()
    df = df.loc[df.def_bin==1]
    X, y = df.drop(['zero_def_bin','del_def_bin','def_bin', 'def_time'],\
            axis=1),df['def_time']
    folds = KFold(n_splits=5)
    cat_params = {'loss_function': 'Quantile:alpha=' + str('0.975'),\
                  'depth': 14, \
                  'learning_rate': 0.4, \
                  'l2_leaf_reg': 5.0,\
                  'eval_metric': 'R2',\
                  'verbose': 200}
    cat_model = CatBoostRegressor(iterations = 100, **cat_params)
    chk = False
    for i, (train_index, test_index) in enumerate(folds.split(X,y)):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        cat_model.fit(X_train,y_train, \
                      eval_set = (X_test,y_test), \
                      use_best_model=True)
        if chk:
            tmp_df = pd.DataFrame(np.ceil(cat_model.get_test_eval()),\
                    index=y_test.index, columns=['lower'])
            y_pred = y_pred.append(tmp_df)

        else:
            y_pred = pd.DataFrame(np.ceil(cat_model.get_test_eval()),\
                                        index=y_test.index,columns=['lower'])
            chk = True
        
    pdb.set_trace()
#    df = pd.read_pickle("./catboost_reg.pk")
    df=df.join(y_pred,how='left')        
       
    return 


def boosted_reg(uni):
    uni = uni.loc[uni.def_bin==1]
    df = uni.loc[uni.def_time<120]
    # Establish Baseline performance.
    # We only provide the required parameters, everything else is default.
    X, y = df.drop(['zero_def_bin','del_def_bin','def_bin', 'def_time'],\
            axis=1),df['def_time']
    folds = KFold(n_splits=3, random_state=None)
    #seed = 1234

    for i, (train_index, test_index) in enumerate(folds.split(X,y)):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        for i in np.linspace(0.1,0.9,9):
#            cat_params = {'loss_function': 'Huber:delta='+str(i),\
            cat_params = {'loss_function': 'Quantile:alpha=' + str(i),\
                          'eval_metric': 'R2',\
                          'verbose': 200}
            cat_model = CatBoostRegressor(iterations = 100,**cat_params)
            para = {'learning_rate': np.linspace(0.1,1,10), \
                    'depth': np.linspace(1,30,30), \
                    'l2_leaf_reg': np.linspace(1,10,10)}
            cat_model.randomized_search(param_distributions=para,\
                                                          X=X_train,y=y_train)
            cat_model.fit(X_train, y_train, eval_set=(X_test,y_test),\
                     cat_features = \
                     X_train.dtypes[X_train.dtypes=='category'].index.tolist(), \
                     use_best_model=True)
            pred = cat_model.predict(X_test)
            print(cat_model.best_score_)

            # === Feature importance === 
#            data = pd.DataFrame({'feature_importance': \
#                                        cat_model.get_feature_importance(), \
#            'feature_names': X.columns}).sort_values(by=['feature_importance'], \
#                                                        ascending=False)
#            data[:20].sort_values(by=['feature_importance'],\
#                ascending=True).plot.barh(x='feature_names', \
#                                                        y='feature_importance')
            # ======
        pdb.set_trace()
    return 
