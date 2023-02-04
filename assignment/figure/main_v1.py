
import sys
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib 
#matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import random

import seaborn as sns
from joblib import delayed, Parallel

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import pyspark.sql.functions as func 
from pyspark.sql.functions import *
from pyspark.sql import types as T
from pyspark import SparkContext

import pickle as pk

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, r2_score
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn import model_selection
import sklearn.pipeline ### REQUIRED FOR KERAS TUNER TO WORK! ###
from sklearn import preprocessing
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV


import six
sys.modules['sklearn.externals.six'] = six
import mlrose

from skgarden import RandomForestQuantileRegressor
from scipy.stats import randint

from scipy import interp


from bayes_opt import BayesianOptimization

import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import catboost as cb

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from imblearn.under_sampling import RandomUnderSampler

import shap

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

        self.X, self.y = under_sample(self.X, self.y, 0.33)

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


def under_sample(X,y,alpha):
    # define undersample strategy
    undersample = RandomUnderSampler(sampling_strategy=alpha)
    X_under, y_under = undersample.fit_resample(X, y)
    return X_under, y_under




def neural_net(df):
    tprs = []
    pres= []
    roc_aucs = []
    prc_aucs=[]
    base_fpr = np.linspace(0, 1, 101)
    # Establish Baseline performance.
    # We only provide the required parameters, everything else is default.
    X_t, y = df.drop(['zero_def_bin','del_def_bin','def_bin'],\
            axis=1),df['def_bin']
    X = X_t[['serv_name','seller_name','credit_score','num_borrower',\
                                        'rem_mth_mat','dti']]
    pdb.set_trace() 
    for i in range (0,10): 
        X_train,y_train = under_sample(X,y,0.33)

#        X_norm = (X-X.mean())/X.std()
        min_max_scaler = preprocessing.MinMaxScaler()
        X_norm = pd.DataFrame(min_max_scaler.fit_transform(X_train))
        
#        seed = 1234
        # Train-test split; Defualt = Stratified.
        X_train, X_test, y_train, y_test = train_test_split(X_norm, y_train, test_size=0.3)
        # Undrsampling performed after train-test split to preserve real world 
        # scenario in test
#        X_train,y_train = under_sample(X_train,y_train,0.1)

#        y_train = pd.DataFrame(pd.get_dummies(y_train).values[:, ::-1].tolist())
        model = Sequential()
        model.add(Dense(90, input_dim=X_train.shape[1], \
                                kernel_regularizer=regularizers.l2(0.000)))
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(BatchNormalization())
        model.add(Dense(X_train.shape[1], \
                                kernel_regularizer=regularizers.l2(0.000)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        # compile the keras model
        opt = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.AUC()])
        # fit the keras model on the dataset
        model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1, validation_split=0.2)
#        model.evaluate(X_test,y_test)
        pred = model.predict(X_test)
#        y_test = pd.DataFrame(pd.get_dummies(y_test).values[:, ::-1].tolist())
        fpr, tpr, thr = metrics.roc_curve(y_test,pred,pos_label=1)
        roc_auc = roc_auc_score(y_test,pred,average=None)
        # calculate model precision-recall curve
        pre, rec, _ = precision_recall_curve(y_test,pred,pos_label=1)
        prc_auc = auc(rec,pre)
        #plt.plot(fpr, tpr, label="ROC fold {}".format(i),alpha=0.3)
        tprs.append(interp(base_fpr, fpr, tpr))
        pres.append(interp(base_fpr, rec, pre))
        roc_aucs.append(roc_auc)
        prc_aucs.append(prc_auc)

        # ======= SHAP analysis =======
#        explainer = shap.KernelExplainer(model.predict,shap.sample(X_train,100))
#        shap_values = explainer.shap_values(shap.sample(X_test,100),nsamples=100)
#        shap.summary_plot(shap_values,X_test,feature_names=X.columns)

        # summarize history for loss
#        plt.plot(history.history['loss'])
#        plt.plot(history.history['val_loss'])
#        plt.title('model loss')
#        plt.ylabel('loss')
#        plt.xlabel('epoch')
#        plt.legend(['train', 'test'], loc='upper left')
#       # pdb.set_trace()
#        plt.show()
        
    pdb.set_trace()
    plot_auc(plt,tprs,base_fpr,roc_aucs)

    return 


def rfq_reg_cv(df):
    df = df.loc[df.def_bin==1]
    X, y = df.drop(['zero_def_bin','del_def_bin','def_bin', 'def_time'],\
            axis=1),df['def_time']
    folds = KFold(n_splits=5)
    rfq_model = RandomForestQuantileRegressor(n_estimators=50,max_leaf_nodes=10,\
            max_depth=6)
    chk=False
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
    df.to_pickle("./rfq_reg.pk")
    pdb.set_trace()
    return



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



def boosted_reg_cv(df):
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
    pdb.set_trace()
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
    df = pd.read_pickle("./catboost_reg.pk")
    df=df.join(y_pred,how='left')        
       
    return 

def boosted_reg(uni):

    uni = uni.loc[uni.def_bin==1]
    df = uni.loc[uni.def_time<120]
    pdb.set_trace()
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
                     cat_features = X_train.dtypes[X_train.dtypes=='category'].index.tolist(), \
                     use_best_model=True)
            pred = cat_model.predict(X_test)
            print(cat_model.best_score_)

#            data = pd.DataFrame({'feature_importance': cat_model.get_feature_importance(), \
#            'feature_names': X.columns}).sort_values(by=['feature_importance'], \
#                                                        ascending=False)
#            data[:20].sort_values(by=['feature_importance'],\
#                ascending=True).plot.barh(x='feature_names',y='feature_importance')

        pdb.set_trace()


    return 


def boosted_model(df,is_cat=False):


    # Establish Baseline performance.
    # We only provide the required parameters, everything else is default.
    X, y = df.drop(['zero_def_bin','del_def_bin','def_bin'],\
            axis=1),df['def_bin']
    #seed = 1234

    X,y = under_sample(X,y,0.33)
    # For K-Fold cross validation
    k = 10 
    folds = StratifiedKFold(n_splits=k, random_state=None)
    tprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)

    gbm_model = lgb.LGBMClassifier(learning_rate=0.1, random_state=42)
    params = {'loss_function': 'Logloss', \
            'eval_metric': 'AUC',\
            'verbose': 200
            }
    cat_model = CatBoostClassifier(iterations = 10, **params)
    for i, (train_index, test_index) in enumerate(folds.split(X,y)):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        if is_cat:
            cat_model.fit(X_train,y_train, \
            cat_features = \
            X_train.dtypes[X_train.dtypes=='category'].index.tolist(), \
            eval_set = (X_test,y_test), \
            use_best_model=True)
            pred = cat_model.predict(X_test, prediction_type='Probability')[:,1]

#            data = pd.DataFrame({'feature_importance': cat_model.get_feature_importance(), \
#              'feature_names': X_train.columns}).sort_values(by=['feature_importance'], \
#                                                            ascending=False)
#            data[:20].sort_values(by=['feature_importance'],\
#                    ascending=True).plot.barh(x='feature_names',\
#                                                    y='feature_importance')
            
        else:
            gbm_model.fit(X_train, y_train, eval_set=[(X_test,y_test), \
                    (X_train,y_train)], verbose=-1,eval_metric='auc')
            pred = gbm_model.predict(X_test, raw_score=True)
            pdb.set_trace()
           
           # Plot Variable Importance
#            data = pd.DataFrame({'Value':gbm_model.feature_importances_,\
#                    'Feature':X_train.columns}).sort_values(by="Value",ascending=False)
#            data[:20].sort_values(by=['Value'],\
#                    ascending=True).plot.barh(x='Feature',\
#                                                    y='Value')
            pdb.set_trace()

        fpr, tpr, thr = metrics.roc_curve(y_test,pred,pos_label=1)
        auc = roc_auc_score(y_test,pred,average=None)
        plt.plot(fpr, tpr, label="ROC fold {}".format(i),alpha=0.3)
        tprs.append(interp(base_fpr, fpr, tpr))
        aucs.append(auc)
        print(roc_auc_score(y_test,pred,average=None))


    plot_auc(plt, tprs,base_fpr,aucs)
    return 


def prep_model_data(df, is_cat=False):

    # Drop: First payment and Maturity date
    df.drop(['first_pmt_date','mat_date'],axis=1,inplace=True)
    
    # Drop variables because of zero variace. 
    df.drop(['prog_ind','prp_val_method', 'io_ind', \
            'amtr_type'],axis=1, inplace=True)

    # Columns that have `object` data types (`dtypes`)
    cols = df.columns[df.dtypes==object]

    # Convert `pre_harp` to 'N' and 'Y' 
    df.loc[df['pre_harp'] != 'N','pre_harp'] = 'Y'

    # Drop loan sequence number and pre-harp
    cols=cols.drop(['loan_seq'])

    df['def_bin'] = np.zeros(len(df))
    df.loc[(df.zero_def_bin==1) | (df.del_def_bin==1),'def_bin'] = 1.0
    df.set_index('loan_seq',inplace=True,drop=True)

    if is_cat:
        cols = df.columns[df.dtypes == 'object']
        df[cols] = df[cols].astype('category')
        df.to_csv('final_data_cat.csv')
        return df
    # Columns that are not ordinal. These can be encoded into `Labels`
    col_label = ['prp_state','seller_name','serv_name','pre_harp']
    # ********** To disable one-hot-code set col_label = col ********
    col_label = cols

    for col in col_label:
        df[col] = df[col].astype('category').cat.codes

    # Columns that are ordinal: these will be one-hot-encoded
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

    df.to_csv('final_data.csv')
    return df 


def handle_missing_data(df):
    '''
    Function to examine columns with empty strings.
    '''
    is_emp = ([])
    for col in df.columns:
        if df[col].isna:
            is_emp.append(col)

    # Handle missing MSA values.
    df.msa.fillna(99999,inplace=True)

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


def plot(df):
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig = plt.figure(figsize=(12, 12))

    # Generate a custom diverging colormap
#    cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
#    # Draw the heatmap with the mask and correct aspect ratio
#    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    col_bar = ['zero_bal_upb','int_upb','act_upb','upb','msa']
    pdb.set_trace()



def plot_auc(plt,tprs,base_fpr,aucs):
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    mean_auc = auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

#    fig = Figure(figsize=(4,4))
    plt.plot(base_fpr, mean_tprs, 'b', alpha = 0.8, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'blue', alpha = 0.2)
    plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha= 0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.grid()
    #plt.axes().set_aspect('equal', 'datalim')
    pdb.set_trace()
    plt.show()

    return

def examine_loan_age(df):
    '''
    Function to identify columns in "monthly performance data", after retaining
    only the rows with `loan age = '000'`, have no data. These columns are 
    excluded from the final data set
    '''
    df = df.replace('', 'null')
    for col in df.columns:
        df.agg(countDistinct(col)).show()
    return

def get_del_def_status(x):
    del_status = set(['3'])
    return 1 - int(set(x).intersection(del_status)==set())

def get_zero_def_status(x):
    del_status = set(['03','06','09'])
    return 1 - int(set(x).intersection(del_status)==set())

def get_data():

    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)
    # ---- Load 2009 Q1 historical data ----
    filepath = '/Users/vashishtha/myGitCode/private/figure/historical_data_2009Q1/historical_data_2009Q1.txt'
    data_rdd = sc.textFile(filepath)
    columns = ['credit_score','first_pmt_date','first_time_buyer','mat_date','msa','mi',\
            'units','ocp_status','cltv','dti','upb','ltv','ori_int_rate','channel',\
            'ppm','amtr_type','prp_state','prp_type','post_code','loan_seq','loan_purpose',\
            'ori_loan_term','num_borrower','seller_name','serv_name','super','pre_harp',\
            'prog_ind','hard_ind','prp_val_method','io_ind']
    # Data: 2009 Q1 historical data
    data = data_rdd.map(lambda x: x.split('|')).toDF(columns)

    # ---- Load 2009 Q1 monthly performance data ----
    filepath = '/Users/vashishtha/myGitCode/private/figure/historical_data_2009Q1/historical_data_time_2009Q1.txt'
    time_rdd = sc.textFile(filepath)
    columns = ['loan_seq','mth_rep_prd','act_upb','loan_del_status','loan_age',\
            'rem_mth_mat','def_setl','mod_flag','zero_bal_cde','zero_bal_dte',\
            'int_rate','def_upb','ddlpi','mi_rec','net_sale','non_mi_rec','exp',\
            'lgl_cost','map_costs','tax_insr','misc_exp','act_loss','mod_cost',\
            'step_mod_flag','def_pmt_plan','eltv','zero_bal_upb','del_acd_int','del_dis',\
            'brw_asst_cde','cmm_cost','int_upb']
    # Data: 2009 Q1 monthly performance data
    time = time_rdd.map(lambda x: x.split('|')).toDF(columns)

    # Two ways of handling of missing `loan age = '000'`: 
    # [1] Remove loans that have missing `loan age = '000'`
    # [2] Get the next minimum loan age, which is `loan age = '001'`
    # Method [1]
#    time_origin = time.filter(time.loan_age=='000')
    # Method [2]
    time_min_loan_age = time.groupBy('loan_seq').agg(func.min('loan_age').\
            alias('min_loan_age'))
    time_origin = time.join(time_min_loan_age, (time.loan_seq == \
            time_min_loan_age.loan_seq) & \
            (time.loan_age==time_min_loan_age.min_loan_age) , "inner").select(time["*"])
    # Examine columns of dataframe that have no data in them.
#    examine_loan_age(time_origin)

    # Compute default status by using `zero balance code`
    zero_bal_def = func.udf(get_zero_def_status,T.IntegerType())
    def_bal_cde = time.groupBy('loan_seq').agg(zero_bal_def(func.collect_list(\
            'zero_bal_cde')).alias('zero_def_bin'))
    # Compute default status by using `Current Loan Delinquency Status`
    del_stat_def = func.udf(get_del_def_status,T.IntegerType())
    def_del_stat = time.groupBy('loan_seq').agg(del_stat_def(func.collect_list(\
            'loan_del_status')).alias('del_def_bin'))

    # Get time to default
    time_len = time.groupBy('loan_seq').count().withColumnRenamed("count","def_time")
    def_len = def_del_stat.join(time_len,time_len.loan_seq==def_del_stat.loan_seq,
            how='inner').select(def_del_stat["loan_seq"],time_len["def_time"])
    def_len_pd = def_len.toPandas()
    pdb.set_trace() 
    # All data without time to default
    universe = \
            data.join(time_origin,data.loan_seq==time_origin.loan_seq, how='left').join(def_bal_cde, \
            data.loan_seq==def_bal_cde.loan_seq, how='left').join(def_del_stat, \
            data.loan_seq==def_del_stat.loan_seq, how='left').select(data["*"], \
            time_origin["act_upb"],time_origin["rem_mth_mat"],time_origin["int_upb"], \
            time_origin["zero_bal_upb"],def_bal_cde["zero_def_bin"], \
            def_del_stat["del_def_bin"])

    pdb.set_trace() 
    # Save data in pickle format.
    universe_pd = universe.toPandas()
#    universe_pd.to_csv('universe_all.csv')
    # Join default time to universe data set
    universe_time_pd=pd.merge(universe_pd, def_len_pd, how="left", on=["loan_seq"])
    # All data with time to default.
#    universe_time_pd.to_csv('universe_all_time.csv')
    return 

def main():
    # Process and Examine data.
#    get_data()
#    universe = pd.read_csv('universe_all.csv')
#    universe.drop('Unnamed: 0',axis=1,inplace=True)
    universe = pd.read_csv('universe_all_time.csv')
#    universe.drop('Unnamed: 0.1',axis=1,inplace=True)
    universe.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1,inplace=True)
    # Examine columns with empty string
    universe_clean = handle_missing_data(universe)
    data = prep_model_data(universe_clean)

#    boosted_reg(data)
#    boosted_reg_cv(data)
    
#    rfq_reg(data)
    rfq_reg_cv(data)
    pdb.set_trace()


#    neural_net(data)
#    data.set_index('loan_seq',inplace=True,drop=True)
#    boosted_model(data,False)

    hyp = hyperOpt(data)
    hyp.hyper_opt_gbm()
    pdb.set_trace()
#    plot(data)

#    catboost_model(universe_clean)
    hyp = hyperOpt(data)
    hyp.hyper_opt_cat()
#    pdb.set_trace()

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()


