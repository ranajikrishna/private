
import sys

import random

import itertools

import csv

from math import *	 # Math fxns.

import xlrd

import pdb
import ipdb
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.pipeline     import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn 		  import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import perceptron

import pickle

from decimal import Decimal

from scipy.optimize import curve_fit
from scipy          import diag, arange, meshgrid, where
from scipy.stats import chi2_contingency
from scipy.stats import chi2, logistic

from random import randint

from pandas.io.pytables import Term
from pandas             import read_hdf
from pandas.io.data	import DataReader

import xlsxwriter
import csv
import statsmodels.api as sm
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

from datetime import date, datetime, timedelta

import jedi

# ==== my prog ===
#import lkdList_example
#import lkdList_practise
#import binTree_node

#import test_pub_pvt1
