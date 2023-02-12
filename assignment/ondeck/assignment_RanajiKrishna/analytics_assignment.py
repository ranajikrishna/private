
'''
Name: Ondeck analytics assignment.

Author: Ranaji Krishna.

Notes:

Please spend some time familiarizing yourself with the data in OnDeck Analytics Assignment.csv. This file contains some interesting information about portfolio performance as of November 1, 2012: Days_delinquent_old represents how many payments a loan has missed as of November 1, 2012 Days_delinquent_new represents how many payments the same loan has missed as of December 1, 2012 The other fields are either categorical or continuous variables describing other things we knew about the loan as of November 1, 2012. Please address the following: 

1) Separate days_delinquent_old and days_delinquent_new into the following groups: (0, 1-5, 5-10, 10-30, 30-60, 60+). Create a transition matrix showing the probability of movement from one group to another. Create another transition matrix showing the probability of movement from one group to another, weighted by outstanding principal balance. 

2) Tell me something interesting about a variable, model, or approach that allows you to distinguish loans whose delinquency is likely to worsen from those whose delinquency is likely to improve.

'''

from myLib import *


def loocv(data, ind_var, dep_var):               # Leave-one-out-cross-validation.
		
	data['predict'] = pd.Series(np.zeros(len(data)),index=data.index) # Predicted log odds.
	for i in data.index:
		reg_data = data.drop(i)		# Drop row with index i.
		model = sm.Logit(reg_data[dep_var], reg_data[ind_var])			   # Logistic Regression model. 
		results = model.fit()							   # Regress.
		data['predict'][i] = results.predict(data.ix[i][ind_var].values.tolist())  # Predict log-odds.
#	store = pd.HDFStore('store.h5')		# Initialize HDF5 variable.	
#	store['data'] = data			# Save data.
	
#	store = pd.HDFStore('store.h5')		# Load HDF5 storage variable.
#	data = store['data']			# Load data.
	data['bin_dep_predict'] = pd.Series(np.zeros(len(data)),index=data.index) 	   # Predicted bin_dep.
	crit_val = pd.DataFrame(np.zeros(400).reshape(100,4),index = np.linspace(0,ceil(max(data['predict'])),100),\
		   columns=['true_pos','false_pos','true_neg','false_neg'])		   # Store values against Critical values.

	# ---- Confusion Matrix Values ---
	for i in crit_val.index:
		data['bin_dep_predict'].where(data['predict']<i, 1, inplace=True)	   # Prediction of 1 (improvement).
		true_pos = np.dot(data['bin_dep'], data['bin_dep_predict'])/sum(data['bin_dep'])	# True positive.
		true_neg = np.dot(1-data['bin_dep'], 1-data['bin_dep_predict'])/sum(1-data['bin_dep'])	# True negative.
		false_neg = 1 - true_pos	# False negative
		false_pos = 1 - true_neg 	# False positive
		crit_val.loc[i] = [true_pos,false_pos,true_neg,false_neg]
		data['bin_dep_predict'] = 0	# Reset to 0.	

	# ---- Plot ROC ---
	plot(crit_val['false_pos'],crit_val['true_pos'])	# Receiver Operating Curve
	plt.ylabel('True Positive Rate (TPR)')
	plt.xlabel('False Positive Rate (FPR)')
	plt.title('Receiver Operating Curve')
	plt.annotate('TPR = 0.71, FPR = 0.28\nCritical Value = 0.162', xy=(0.284274, 0.71875), xytext=(0.15, 0.85),
            	     arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2")
            )
	plt.grid(True)
	plt.show()

	return(0)
		

def log_reg(data):	       # Perform Logistic Regression.

	data['bin_dep'] = pd.Series((data['days_delinquent_old'] - data['days_delinquent_new']),index=data.index) # Delinquent (Old - New).
	#data['intercept'] = 1				# Add intercept.
	data = data[data['bin_dep']!=0]			# Remove loans with delinquent (Old = New). 
	data.loc[data['bin_dep']<0, 'bin_dep'] = 0	# Set worse delinquencies (-ve values) to 0.
	data.loc[data['bin_dep']>0, 'bin_dep'] = 1	# Set improved delinquencies (+ve values) to 1.

	data['type_bin'] = 0				# Dummy variable (= 0 for type = 'Loan - Renewal').	
	data.loc[data['type']=='Loan - New Customer', 'type_bin'] = 1	# Dummy variable (= 1 for type = 'Loan - New Customer').

	data['sales_bin'] = 4				# Sales_channel bin values (= 4 for sales_channel = 'Promonotory').	
	data.loc[data['sales_channel__c'] == 'FAP: Managed Application Program', 'sales_bin'] = 1	# FAP = 1.
	data.loc[data['sales_channel__c'] == 'Referral', 'sales_bin'] = 2				# Refferal = 2.
	data.loc[data['sales_channel__c'] == 'Direct' , 'sales_bin'] = 3				# Direct = 3.

	data['collection_bin'] = 3			# Sales_channel bin values (= 3 for collection_menthod = 'Promonotory').	
	data.loc[data['current_collection_method'] == 'ACH Pull', 'collection_bin'] = 1			# ACH Pull = 1.
	data.loc[data['current_collection_method'] == 'Split Funding', 'collection_bin'] = 2		# Split Funiding = 2.

	ind_var = ['new_outstanding_principal_balance', 'initial_loan_amount',\
		  'term', 'sales_bin']			# Independent variables.
	
	dep_var = ['bin_dep']				# Dependent variable.
	model = sm.Logit(data[dep_var], data[ind_var])  # Logistic Regression model.

	results = model.fit()				# Regress.
	loocv(data, ind_var, dep_var)			# Leave-one-out-cross-validation.

	return(results,ind_var)

def tran_mat(data,cat):		# Forms transition prob. matrix and weighted tran. prob. matrix .

	tran_mat = pd.DataFrame(np.zeros((len(cat),len(cat))),index=cat,columns=cat)	       # Transition matrix.
	wgt_mat = pd.DataFrame(np.zeros((len(cat),len(cat))),index=cat,columns=cat)	       # Weight matrix.

	for alpha in cat:
		subset = data.query('days_delinquent_old_cat==@alpha')          # days_delinquent_old_cat = category. 
		if subset.empty: continue					# If no-mapping then skip.
		freq_count = subset.groupby('days_delinquent_new_cat').count().iloc[:,1]       # Transition freq. 
		tran_mat.loc[alpha] = freq_count[cat].fillna(0)/sum(freq_count[cat].fillna(0)) # Populate transition matrix.

		weights = subset.groupby('days_delinquent_new_cat')['new_outstanding_principal_balance'].sum() \
			  /sum(subset['new_outstanding_principal_balance'])     # Compute weights.
		wgt_mat.loc[alpha] = weights[cat].fillna(0) 			# Populate weights matrix.

	tran_mat_wgt = (tran_mat * wgt_mat).div((tran_mat * wgt_mat).sum(1),axis = 'rows')    # Weighted Tran. matrix.

	return(tran_mat, tran_mat_wgt)


def group(data, col, cat):	# Groups days into categories.

	data[col+'_cat'] = pd.Series(np.random.randn(len(data[col])), index=data.index)       # Add column of categories. 
	data.at[(data[col]==0), col+'_cat']=cat[0]				# 0

	data.at[(data[col]<=5) & (data[col]>=1), col+'_cat']= cat[1]		# 1- 5
	data.at[(data[col]<=10) & (data[col]>=5), col+'_cat']= cat[2]		# 5 - 10
	data.at[(data[col]<=30) & (data[col]>=10), col+'_cat']= cat[3]      	# 10 - 30
	data.at[(data[col]<=60) & (data[col]>=30), col+'_cat']= cat[4]	        # 30 - 60
	data.at[(data[col]>=60), col+'_cat']= cat[5]				# 60+

	return(data)

def main (argv = None):


	# ---- Get data ----
	file_location = '/Users/vashishtha/myGitCode/onDeck/all_data.xlsx'	# File location path.
	workbook = xlrd.open_workbook(file_location)
	sheet = workbook.sheet_by_index(0)					# Import data from Excel.

	date_col = [0,12]	# Columns with date.
	universe = [[xlrd.xldate_as_tuple(sheet.cell_value(r,c), workbook.datemode)[0:3] \
	           if c in date_col and r!=0 else sheet.cell_value(r,c) \
	           for c in range(sheet.ncols)] for r in range (sheet.nrows)]	           # Read data.
	
	universe = pd.DataFrame(universe, dtype = 'd', columns=list(universe[0])).ix[1:]   # Convert to dataframe.

	# ==== Exercise 1 ==== 
	sel_col = ['days_delinquent_old', 'days_delinquent_new', 'new_outstanding_principal_balance']     # Select columns.
	rel_data = universe[sel_col]  	   # Relevant data.
	cat = ['A','B','C','D','E','F']    # Define categories.
	
	col = 'days_delinquent_old'	   # Column old delinquent.
	rel_data = group(rel_data,col,cat) # Group old days into categories.
	col = 'days_delinquent_new'	   # Column new delinquent.
	rel_data = group(rel_data,col,cat) # Group new days into categories.

	rel_data = rel_data.convert_objects(convert_numeric=True).dropna() # Drop rows with missing values.

	trp_mat, trp_wgt = tran_mat(rel_data,cat)   # Tran. Prb. Matrix, Weighted Tran. Prb. Matrix.
	'''
	writer = pd.ExcelWriter('prb_matrices.xlsx', engine='xlsxwriter')  # Write to xlsx.
	trp_mat.to_excel(writer,'Sheet1')
	trp_wgt.to_excel(writer,'Sheet2')
	writer.save()				   # Save to xlsx.
	'''

	# ==== Exercise 2 ====
	sel_col = ['days_delinquent_old', 'days_delinquent_new', 'lender_payoff', 'average_bank_balance__c', \
		   'new_outstanding_principal_balance', 'initial_loan_amount', 'term', 'fico','type',\
		   'sales_channel__c','current_collection_method']     # Select columns.
	rel_data = universe[sel_col]  	   # Relevant data.
	rel_data = rel_data.convert_objects(convert_numeric=True).dropna() # Drop rows with missing values.
	
	results, var = log_reg(rel_data)
	#print (results.summary())
	#print (np.exp(results.params))
	#data['predict'][i] = model.predict

	return (0)

if __name__ == '__main__':
	status = main()
	sys.exit(status)


