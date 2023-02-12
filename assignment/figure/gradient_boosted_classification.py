
import pandas as pd
import numpy as np
import random

import prep_model_data as pmd
import examine_model as em
import tag_rare as tg

import multiprocessing as mp

from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold 
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import catboost as cb
from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

import sampling as smp      # Self written code for undersampling
import all_plot as ap   # Self written code for plotting auc

from scipy.stats import randint
from scipy import interp

import matplotlib 
#matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import pdb
import sys 


def parallel_clf(df,n_splits,proc,is_cat=False):
    # Establish Baseline performance.
    # We only provide the required parameters, everything else is default.
    X, y = df.drop(['zero_def_bin','del_def_bin','def_bin'],\
            axis=1),df['def_bin']
#    seed = 1234

#    X,y = smp.under_sample(X,y,0.33)  # Perform undersampling.
    base_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []

    params = {'loss_function': 'Logloss', \
            'eval_metric': 'AUC',\
            'early_stopping_rounds':5,\
            'verbose': 200
            }
    folds = StratifiedKFold(n_splits=n_splits,shuffle=True,\
                                                        random_state=None)
    gbm_model = lgb.LGBMClassifier(learning_rate=0.1)
    cat_model = CatBoostClassifier(iterations = 10, **params)

    for i, (train_index, test_index) in enumerate(folds.split(X,y)):
        X_tr, X_te = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        # Target encode after splitting train and test. 
        X_train, X_test = pmd.target_encoder(X_tr,y_train), \
                                        pmd.target_encoder(X_te,y_test)
        if is_cat:
            # CatBoost
            cat_model.fit(X_train,y_train, \
            cat_features = \
            X_train.dtypes[X_train.dtypes=='category'].index.tolist(), \
            eval_set = (X_test,y_test), \
            use_best_model=True)
            pred = cat_model.predict(X_test, \
                                        prediction_type='Probability')[:,1]

            # === Variable Importance ===
#            data = pd.DataFrame({'feature_importance': \
#                               cat_model.get_feature_importance(), \
#                           'feature_names': X_train.columns}).sort_values(by= \
#                           ['feature_importance'], ascending=False)
#
#            data[:20].sort_values(by=['feature_importance'],\
#                    ascending=True).plot.barh(x='feature_names',\
#                                                    y='feature_importance')
            # ======
        
        else:
            # Light GBM
            gbm_model.fit(X_train, y_train, eval_set=[(X_test,y_test), \
                    (X_train,y_train)], verbose=-1,eval_metric='auc')
            pred = gbm_model.predict(X_test, raw_score=True)
           
            # === Variable Importance ===
#            data = pd.DataFrame({'Value':gbm_model.feature_importances_,\
#                    'Feature':X_train.columns}).sort_values(by= \
#                    "Value",ascending=False) 
#            data[:20].sort_values(by=['Value'],\
#                            ascending=True).plot.barh(x='Feature',y='Value')
            # ======

        fpr, tpr, thr = metrics.roc_curve(y_test,pred,pos_label=1)
        roc = pd.DataFrame({'tpr':tpr, 'fpr':fpr,'thr':thr})
        # Examine performance of model
#        em.examine_model(X_te,y_test,X_tr,y_train,pred,-3.167001)
        auc = roc_auc_score(y_test,pred,average=None)
#        plt.plot(fpr, tpr, label="ROC fold {}".format(i),alpha=0.3)
        tprs.append(interp(base_fpr, fpr, tpr))
        aucs.append(auc)
    print(aucs)
    return [aucs,tprs]


def boosted_clf(df,is_cat=False):


#    ndf = tg.create_rare_tag(df)


    # For K-Fold cross validation
    n_repeats = 3
    n_splits = 10 

    repeat_auc = []

    #pool = mp.Pool(mp.cpu_count())
    pool = mp.Pool(processes=4)
    repeat_auc.append(pool.starmap(parallel_clf,[[df,n_splits,i,False] \
                                                for i in range(n_repeats)]))
    pool.close()
    ap.repeatedKFold_plot(repeat_auc[0])
    pdb.set_trace()

    repeat_auc.append(aucs)
    print(roc_auc_score(y_test,pred,average=None))
    
    
    pdb.set_trace()
    ap.plot_auc(plt,tprs,base_fpr,repeat_auc)
    return 
