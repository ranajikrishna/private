
import pandas as pd
import numpy as np
import random

from bayes_opt import BayesianOptimization

from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import catboost as cb

import lightgbm as lgb

import sampling as smp      # Self written code for undersampling

import sys
import pdb

class hyperOpt():
    '''
    Bayesian Optimization for Hyper parameter tuning.
    '''

    def __init__(self,df):
        self.seed = 1234
        self.test_size = 0.3


        self.X, self.y = df.drop(['zero_def_bin','del_def_bin','def_bin'],\
                axis=1),df['def_bin']

        self.X, self.y = smp.under_sample(self.X, self.y, 0.33)

#        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(\
#                self.X, self.y, test_size=self.test_size, random_state=self.seed)
        self.X_train, self.y_train = self.X, self.y


    def lgb_classifier(self, num_leaves, max_depth, lambda_l2, \
            lambda_l1, min_data_in_leaf, learning_rate):
        params = {"boosting_type": 'gbdt', \
                "objective" : "binary", \
                "metric" : "auc", \
                "is_unbalance": True, \
                "num_boost_round": 1000, \
                "early_stopping_rounds": 100, \
                "feature_fraction": 0.5, \
                "bagging_fraction": 0.5, \
                "bagging_seed" : 42, \
                "num_threads" : 20, \
                "verbosity" : -1,\
                "learning_rate" : float(learning_rate), \
                "num_leaves" : int(num_leaves), \
                "max_depth" : int(max_depth), \
                "lambda_l2" : lambda_l2, \
                "lambda_l1" : lambda_l1, \
                "min_data_in_leaf": int(min_data_in_leaf) \
                }
        train_data = lgb.Dataset(self.X_train, self.y_train,params={'verbose': -1})
        cv_result = lgb.cv(params, \
                train_data, \
                1000, \
                #early_stopping_rounds=10, \
                stratified=True, \
                verbose_eval=False,\
                nfold=5)
        return cv_result['auc-mean'][-1]

    def cat_classifier(self, depth, l2_leaf_reg, num_boost_round, learning_rate, max_leaves):
        params = {
                "loss_function": "Logloss",
                "eval_metric" : "AUC", 
                "depth" : int(depth),
                "l2_leaf_reg" : int(l2_leaf_reg),
                "learning_rate" : float(learning_rate),
                #"max_leaves": int(max_leaves),
                "random_state" : 42,
                "logging_level" : "Silent",
                "thread_count": 24,
                "num_boost_round": 1000
                }

        categorical_features = self.X_train.dtypes[self.X_train.dtypes=='category'].index.tolist()
        train_data = catboost.Pool(data=self.X_train, label=self.y_train, cat_features=categorical_features)
        cv_result = catboost.cv(train_data,
               params,
               #early_stopping_rounds=100,
               stratified=True,
               nfold=3)
        return cv_result['test-AUC-mean'].iloc[-1] 


    def hyper_opt_gbm(self):
        lgbBO = BayesianOptimization(self.lgb_classifier, {\
               'num_leaves': (10, 50),\
               'max_depth': (5, 50),\
               'learning_rate':(1e-6,1), \
               'lambda_l2': (1.5, 3),\
               'lambda_l1': (1.5, 3),\
#               'min_child_samples': (50, 100),\
               'min_data_in_leaf': (10, 100) \
#               'early_stopping_rounds': 100\
#               'num_boost_round': (1000),\
                })

        opt = lgbBO.maximize(n_iter=13, init_points=2)
        pdb.set_trace()

        return


    def hyper_opt_cat(self):
        catBO = BayesianOptimization(self.cat_classifier, { \
               'learning_rate': (0.2,1),\
               'depth': (1, 4), \
               'l2_leaf_reg': (2,5), \
               'num_boost_round': (100, 200), \
               'max_leaves': (50,100) \
               })

        catBO.maximize(n_iter=8, init_points=2)
        pdb.set_trace()

        return 
