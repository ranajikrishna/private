#
# Name: Final, Question 12.
#
# Author: Ranaji Krishna.
#
# Note:

import sys;
import numpy as np;
import random;
import itertools;
from math import *;	 # Math fxns.
import pandas as pd;
import xlrd;

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from decimal import Decimal
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from random import randint
from cvxopt import blas, lapack, matrix, solvers
solvers.options['show_progress'] = 0
from scipy import linalg as LA
from numpy.linalg import matrix_rank

import svmutil as svm



def supVector(data, C):

	options = '-s 0 -t 1 -d 2 -r 1 -g 1 -c ' + str(C); # Set Options.

	prob = svm.svm_problem(np.array(data[2]).tolist(), np.array(data.loc[:,0:1]).tolist());	# Set problem.

	model = svm.svm_train(prob, options);		   # Call libsvm.
	
	return(0);



def main(argv = None):

	file_location = '/Users/vashishtha/myGitCode/ML/homeWorks/final/que11_data.xlsx';	 # File location path.
	workbook = xlrd.open_workbook(file_location);
	sheet = workbook.sheet_by_index(0);

	data_train = [[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range (sheet.nrows)]; # Import in-sample data from Excel.
	
	data_train =  pd.DataFrame(data_train);

	C = 1e6;		  # Setting a hard boundary.
	supVector(data_train, C); # Support vector machine evaluation. 

	return(0);


if __name__ == '__main__':
	status = main();
	sys.exit(status);
