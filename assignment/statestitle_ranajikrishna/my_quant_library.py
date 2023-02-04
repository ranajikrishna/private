
import sys

import random

import csv

from math import *	 # Math fxns.

import functools 

import time

import h2o
#from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch


#import xlrd
import pdb

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn 		  import datasets
from sklearn.pipeline     import Pipeline
from sklearn.neighbors import *
from sklearn.decomposition import PCA
from sklearn.svm import SVC as SVC
from sklearn.preprocessing import normalize, scale
from sklearn import preprocessing 

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#from imblearn.over_sampling import SMOTE	# Oversampling technique
#from imblearn.under_sampling import NearMiss	# Undersampling

from statsmodels import regression
import statsmodels.api as sm

import pickle

from decimal import Decimal

from scipy.optimize import curve_fit
from scipy          import diag, arange, meshgrid, where
from scipy import interp
import scipy as sp

from random import randint

from pandas.io.pytables import Term
from pandas             import read_hdf
from pandas.io.data	import DataReader

import xlsxwriter
import pandas as pd

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.structure 		 import LinearLayer, SigmoidLayer
from pybrain.structure 		 import FeedForwardNetwork
from pybrain.structure 		 import FullConnection

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot

from numpy.random import multivariate_normal
import numpy as np

#from mpldatacursor import datacursor
from datetime import date, datetime, timedelta

from joblib import Parallel, delayed
import multiprocessing

#import jedi

# Hyper-optimization 
import itertools as it
# Hyperoptimization using Bayesian optimization
#from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline
from functools import partial

# From R (.rds) to Python
#import rpy2.robjects as robjects 

# Computes impact of ARPA and Retention.
#import model_comparison as mc		# Compare models.
#import pick_mod as pm			# Pick models.	
#import cat_var as cv			# Categorical Variables. 

#import fxn_conv as fc 

import psycopg2 # Read SQL from Python.

from statsmodels.stats.outliers_influence import variance_inflation_factor


