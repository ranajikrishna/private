#
# Name: Final, Question 11 - 12
#
# Author: Ranaji Krishna.
#
# Note:
# Consider the following training set generated from a target function 
# f : X -> {-1, +1} where X = R^2 x1 = (1,0), y_1 = -1 x_2 = (0,1), y_2 = -1 x_3 = (0,-1), 
# y_3 = -1, x_4 = (-1,0), y_4 = +1 x_5 = (0,2), y_5 = +1, x_6 = (0,-2), y_6 = +1
# x_7 = (-2,0), y_7 = +1
# Transform this training set into another two-dimensional space Z
# z_1 =x_2 -2x_1-1, z_2 =x^2_1-2x_2 + 1
#
# Question 11:
# Using geometry (not quadratic programming), what values of w (without w0) and b 
# specify the separating plane w^Tz + b = 0 that maximizes the margin in the Z space? 
# The values of w1, w2, b are:
# -1,1,-0.5; 1,-1,-0.5; 1,0,-0.5; 0,1,-0.5 
#
# Question 12:
# Consider the same training set of the previous problem, but instead of explicitly transforming 
# the input space X, apply the hard-margin SVM algorithm with the kernel
#			K(x, x') = (1 + x^Tx')^2
# (which corresponds to a second-order polynomial transformation). Set up the expression 
# for L(\alpha1...\alpha7) and solve for the optimal \alpha1, ..., \alpha7 (numerically, using a quadratic 
# programming package). The number of support vectors you get is in what range?#
#


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

def rbf_regular(data, n_samples, C, G):


	options = '-s 0 -t 2 -g ' + str(G) + ' -q -c ' + str(C); 			   # Set Options.
	prob = svm.svm_problem(np.array(data[2]).tolist(), np.array(data.loc[:,0:1]).tolist());	# Set problem.
	model = svm.svm_train(prob, options);				   # Call libsvm.

	[labels, accuracy, values] = svm.svm_predict(np.array(data[2]).tolist(), np.array(data.loc[:,0:1]).tolist(), model, '-q');

	return(accuracy[0]);

def svm_algo(data, gamma, n_samples):
 	
	# --- Support Vector Machines ---
#tmp = [exp(np.dot(data.loc[x,0:1] - data.loc[y,0:1], data.loc[x,0:1] - data.loc[y,0:1])) for x in data.index for y in data.index];
	tmp = [exp(-1 * gamma * LA.norm(data.loc[x,0:1] - data.loc[y,0:1])**2) for x in data.index for y in data.index];	
	
	M = matrix(tmp,(100,100));
#M = m * m.T;	
 	y = matrix(data.loc[:,2].as_matrix(), tc = 'd');
        Y = y * y.T;	
	P = matrix(np.multiply(M,Y), tc = 'd');
	q = matrix(-1 * np.ones(n_samples), tc = 'd');
	A = matrix(data[2], tc = 'd');
	b = matrix(np.zeros(1), tc = 'd');
	G1 = matrix(np.diag(np.ones(n_samples) * -1), tc = 'd');
	G2 = matrix(np.diag(np.ones(n_samples)), tc = 'd');
	G  = matrix([[G1,G2]]);
	h1 = matrix(np.zeros(n_samples), tc = 'd');
	h2 = matrix(np.ones(n_samples) * float("inf"), tc = 'd');
	h  = matrix([[h1,h2]]);
	
	# ---- Convex Optimization ----
	solvers.options['abstol'] = 1e-9;	 
	solvers.options['reltol'] = 1e-8;	 
	solvers.options['feastol'] = 1e-9;	 
	
	sol = solvers.qp(P,q,G1,h1,A.T,b);	# Run convex optimisation.
	alpha = np.ravel(sol['x']);		# Alpha's.
	position = np.where(alpha>1);		# All \alpha > 0.
	max_itr = np.argmax(alpha);		# \alpha chose to calculate intercept (b).
	# -----------

	w = sum(np.multiply((data.values[position,2]*alpha[position]).T,data.values[position,0:2])[0][:]);

	c =  data.values[max_itr,2]- np.dot([exp(-1 * gamma * np.linalg.norm((data.values[position,0:2] - data.values[max_itr,0:2])[0][i])**2) for i in range(0,np.size(position))], (data.values[position,2]*alpha[position]).T);

#c = data.loc[max_itr, 2] - np.dot(w, data.loc[max_itr, 0:1].T);


	data[3] = np.sign(np.dot(w, data.loc[:,0:1].T) + c);	# Estimate value using SVM, True.

#for j in range(0,n_samples):
#		data.loc[j,3] = np.sign(np.sum(data.values[position,2] * position * [exp(-1 * gamma * np.linalg.norm(data.loc [j,0:1] - data.values[position[0][i],0:2])**2) for i in range(0,np.size(position))]) + c);
	
	data[4] = np.where(data[2] != data[3],1,0);	        # SVM Classification Verificaion.	
	
	return(float(sum(data[4]))/n_samples);

def main(argv = None):

	n_samples = 100;

	inSample = pd.DataFrame(np.ndarray((n_samples,4), dtype = float));	# Matrix to store values (cols: x-values, y-values, f(x,y), identification).

	n_iter = 10;
	G = 1.5; C = 1e6;
	error_svm = pd.DataFrame(np.ndarray((n_iter,1), dtype = float));
	error_rbf = pd.DataFrame(np.ndarray((n_iter,1), dtype = float));
	
	for i in range(0, n_iter):
		inSample[0] = np.random.uniform(-1,1,n_samples);	# x1-values. 
 		inSample[1] = np.random.uniform(-1,1,n_samples);	# x2-values.

		inSample[2] = [np.sign(inSample.loc[x,1] - inSample.loc[x,0] + 0.25 * sin(pi * inSample.loc[x,0])) for x in inSample.index];

#error_svm.loc[i,0] = svm_algo(inSample, 1.5, n_samples);	
		error_rbf.loc[i,0] = rbf_regular(inSample, n_samples, C, G);


	t = 9;	
	return(0);



if __name__ == '__main__':
	status = main();
	sys.exit(status);
