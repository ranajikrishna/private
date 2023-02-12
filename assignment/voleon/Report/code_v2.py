
# ******************************
# Name: code_v2.py
# Auth: Ranaji Krishna
# Date: Feb. 25 2016	
# Function: 
# 	get_data: Load csv files.
# 	regress:  Carry out optimization. 
# 	plot_reg: Plot data	
# 	estimator:Try out diferent estimators
# **************************

import pdb
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import statsmodels.api as sm



def get_data():

	file_location = '../data_1_1.csv'
	data = pd.read_csv(file_location)
	return(data)

def regress (mod, data):

	# Optimization. Note: HC3 covariance estimator used.
	res_mod = mod.fit(cov_type='HC3')
	#print res_mod.summary()

	# --- White test ----
	test = sm.stats.diagnostic.het_white(res_mod.resid, \
		res_mod.model.exog, retres=False)

	name = ['F statistic', 'p-value']
 #	print lzip(name, test)
	
	return

def plot_reg(plot_lines, data):

	# Scatter plots of x-, y- data points
	plt.scatter(data.x,data.y)
	first_legend = plt.legend(plot_lines[0],["ols","wls","fGLS","QuantReg"], loc=1)

	plt.ylabel('y')
	plt.xlabel('x')
	plt.title('data_1_5: OLS-, WLS-, fGLS- , Quantile- Regression Fit')
	return

def estimator (data):

	# ---- OLS ----
	mod_ols = sm.OLS(data.y, data[['x','const']])
	regress(mod_ols, data)
	# Plot OLS fit
	l1, = plt.plot(data.x, mod_ols.fit().fittedvalues,'b-', label="ols")

	# ---- Feasible GLS ----
	resid = sm.OLS(data.y, data[['x','const']]).fit().resid
	predict_res = sm.OLS(np.log(resid**2), \
			data[['x','const']]).fit().fittedvalues

	mod_fgls = sm.regression.linear_model.WLS(data.y, \
		data[['x','const']], weights=1./np.exp(predict_res))

	regress(mod_fgls, data)
	# Plot fGLS fit
	l2, = plt.plot(data.x, mod_fgls.fit().fittedvalues,'r-')

	# ---- Weighted LS ----
	mod_wls = sm.regression.linear_model.WLS(data.y, \
			data[['x','const']], weights=1./data.x**2)

	regress(mod_wls, data)
	# Plot WLS fit
	l3, = plt.plot(data.x, mod_wls.fit().fittedvalues,'g-')

	# ---- Quantile Regression ----
	mod_qnt=sm.QuantReg(data.y,data[['x','const']])
	# ---- Model Selection --
	qtile = [0.25, 0.45, 0.75, 0.95]
	for q in qtile:
		mod_res=mod_qnt.fit(q)
		#print mod_res.summary2()	
	#print mod_qnt.fit(q=0.45, cov_type='HC3').summary2()
	# Plot QR fit
	l4, = plt.plot(data.x, mod_qnt.fit(q=0.45).fittedvalues,'k-')
	
	plot_lines=[]
	plot_lines.append([l1, l2, l3, l4])
	plot_reg(plot_lines, data)
	return
	

def main(argv=None):

    data = get_data()
    pdb.set_trace()
    # Add ones for intercept.
    data = sm.add_constant(data, prepend=False)
    estimator(data)		# Try different estimators . 
    return

if __name__ == '__main__':
	status = main()
	sys.exit(status)
