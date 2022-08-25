
import sys

import random

import csv

from math import *	 # Math fxns.

import functools 


import xlrd
import pdb

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn 		  import datasets
from sklearn.pipeline     import Pipeline

from statsmodels import regression
import statsmodels.api as sm
from statsmodels.compat import lzip

import pickle

from decimal import Decimal

from scipy.optimize import curve_fit
from scipy          import diag, arange, meshgrid, where
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

from mpldatacursor import datacursor
from datetime import date, datetime, timedelta

from joblib import Parallel, delayed
import multiprocessing

#import jedi

import itertools as it




