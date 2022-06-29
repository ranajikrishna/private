
import sys

import random

import itertools

import csv

from math import *	 # Math fxns.

import xlrd

import ipdb
import pdb

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.pipeline     import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn 		  import datasets
from sklearn.linear_model import LogisticRegression

from statsmodels import regression
import statsmodels.api as sm

from yahoo_finance import Share

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

from mpldatacursor import datacursor
from datetime import date, datetime, timedelta

import jedi

import itertools as it

# ==== my quant fxns. ===
import my_get_price as mgp		# Gets prices from Yahoo.
import my_similarity_score as mss	# Compute similarity score of ordering.
import my_trade_sig as mts
import momt_ret 
import my_plot				# Plot asset and trade signals.
import my_hedge_sig as mhs		# Build hedging strategy.
import max_drawdown as mdd

# ==== my prog ===
#import lkdList_example
#import lkdList_practise
#import binTree_node
#import test_pub_pvt1



