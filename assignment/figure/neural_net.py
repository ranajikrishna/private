
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from imblearn.under_sampling import RandomUnderSampler

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, r2_score
from sklearn import metrics

from scipy import interp

import shap

import sampling as smp  # Self written code for undersampling
import all_plot as ap   # Self written code for plotting auc

import sys
import pdb

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
        X_train,y_train = smp.under_sample(X,y,0.33)

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
        model.compile(loss='binary_crossentropy', optimizer=opt, \
                                            metrics=[tf.keras.metrics.AUC()])
        # fit the keras model on the dataset
        model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1, \
            validation_split=0.2, callbacks=EarlyStopping(monitor='val_loss'))
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
#        pdb.set_trace()
#        plt.show()
        
    pdb.set_trace()
    ap.plot_auc(plt,tprs,base_fpr,roc_aucs)

    return 
