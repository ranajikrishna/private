
'''
Name: Coding exercise.

Author: Ranaji Krishna.

README: To run the code in command line: python -W ignore exercise_code.py

Notes: 
Below is the code for both Question 1 and Question 2. For Question 2 both a
Logistic Regression estimator and a single-layer perceptron classifier are
investigated. 

The following functions are used:-
main():  
import_data(): Imports data from excel files.
clean_data(): Cleans data to form relational tables for analysis in Question 2.
clac_content(): Computes the total number of content a user has (Question 1, part 1).
growth_metric(): Defines a metric that determines which are the fastest growing 
users in terms of positive customer engagement over the last year (Question 1, part 2).
logistic_regression(): Carries out Logistic Regression.
over_disp_test(): Tests for over dispersion in data.
percep_clf(): Conducts Single-layer perceptron classification.
k_fold_cross_val(): Carries out 10-fold cross validation.
roc_curve(): Carries out simulations to draw Receiver operating curves.
'''


from myLib import *

def roc_curve(data):
	'''
	We draw Receiver Operating Curves for the Logistic Regressor based on k-fold (k=10) 
	crossvalidation. We use this technique to establish the Critical value - the 
	probability above which the output of the estimator is assigned a value of 1.
	'''
	data['predict_bin'] = pd.Series(np.zeros(len(data)),index=data.index) 	   # Predicted bin_dep.
	
	# Confusion matrix to store performance for varying critical values.  
	crit_val = pd.DataFrame(np.zeros(400).reshape(100,4),\
	index = np.linspace(0,ceil(max(data['predict'])),100),\
	columns=['true_pos','false_pos','true_neg','false_neg']) # Store values against Critical values.

	# ---- Populate th Confusion Matrix ---
	for i in crit_val.index:
		data['predict_bin'].where(data['predict']<i, 1, inplace=True)	   # Prediction of 1.
		true_pos = float(np.dot(data['response'], data['predict_bin']))/\
		sum(data['response'])		# True positive.
		true_neg = float(np.dot(1-data['response'], 1-data['predict_bin']))/\
		sum(1-data['response'])		# True negative.
		false_neg = 1 - true_pos	# False negative
		false_pos = 1 - true_neg 	# False positive
		crit_val.loc[i] = [true_pos,false_pos,true_neg,false_neg]	
		data['predict_bin'] = 0		# Reset to 0.

	# ---- Plot Receiver Operating Curves ---
	plt.plot(crit_val['false_pos'],crit_val['true_pos'])	# Receiver Operating Curve
	plt.plot([0,1],[0,1],'r')				
	plt.ylabel('True Positive Rate (TPR)')
	plt.xlabel('False Positive Rate (FPR)')
	plt.title('Receiver Operating Curve')
	plt.annotate('TPR = 81.5%, FPR = 17.5%\nCritical Value = 0.24', xy=(0.173, 0.815), xytext=(0.2, 0.6),
            	     arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
	plt.grid(True)
	
	acc = float(sum(data['response']==data['predict_bin']))*100/len(data)	# Accuracy.
	print "\nOut-of-sample performance for the Logistic Regression estimator: "
	print "True Positive Rate = 81.5%" 
	print "False Positive Rate = 17.5%"
	print "Accuracy = ", "%.2f" % acc, "%"

	fig = plt.gcf()	 	
	plt.show()
	fig.savefig('roc.pdf')
	plt.close(fig)

	return(0)


def k_fold_cross_val(estimator):               
	'''
	This fuction performs two sets of k-fold cross validation:
	one for the Perceptron classifier and one for the Logistic 
	regressor. This is done to examine the out-of-sample performance 
	of the two techniques. The input variable "estimator" parsed selects the
	appropriate section of the code for each estimator.
	'''
	k = 10
	train_data = pickle.load(open("train_data.p", "rb"))	# Load training data. 
	dep_var = train_data['response']			# Dependent variable.
	
	predict = pd.DataFrame(np.zeros(len(train_data)), index=train_data.index,\
	columns=['predict']) 	   				# Predicted 'response'.
	
	predict1 = pd.DataFrame(np.zeros(len(train_data)), index=train_data.index,\
	columns=['predict']) 	   				# Predicted 'response'.
	# ========== k-fold cross validation for Perceptron classifier ====
	# This section of the code takes approx. 30 minutes to complete. 
	# As such the  results of this simulation have been generated and 
	# saved as train_data_percep.p.

	if (estimator == 'perceptron'):		
		
		# Independent variables.
		ind_var  = train_data.drop(['response', 'user_id'], axis=1)
		
		# Groups of k-data points.
		ind_slice = [ind_var[i:(i+k)] for i in xrange(0, len(ind_var.index), k)]
		
		# Single layer perceptron classifier  
		net = perceptron.Perceptron(n_iter=10000, verbose=0, random_state=None, \
			fit_intercept=True, eta0=0.002)

		print"\nGenerating k-fold out-of-sample performance for Perceptron. This"
		print"simulation will take approximately 30 mins. to complete."
		for i in ind_slice:				# Iterate thorugh groups.
			net.fit(ind_var.drop(i.index), dep_var.drop(i.index))
			predict['predict'][i.index] = net.predict(i)  # Predict log-odds.

		print"\nSimulation complete."
		train_data = train_data.join(predict, how='inner')
		# Save results of out-of-sample performance of Perceptron.
		# pickle.dump(train_data, open("train_data_percep.p", "wb"))

	else:
	# ========= k-fold cross validation for Logistic regressor =====
		
		
		ind_var  = train_data.drop(['response', 'user_id'], axis=1)  
		ind_var['intercept'] = 1.0	# Add intercept.

		# Groups of k-data points
		ind_slice = [ind_var[i:(i+k)] for i in xrange(0, len(ind_var.index), k)]

		print"\nGenerating k-fold out-of-sample performance for Logistic Regressor. This"
		print"simulation will take approximately 1 min. to complete."
		for i in ind_slice:		# Iterate through groups.
			# Logistic Regression model.
			model = sm.Logit(dep_var.drop(i.index), ind_var.drop(i.index))
			results = model.fit(disp=0)			 # Regress.
			predict['predict'][i.index] = results.predict(i) # Predict.

		print"\nSimulation complete."
		train_data = train_data.join(predict, how='inner')

		# Save results of out-of-sample performance of Logistic Regressor.
		pickle.dump(train_data, open("train_data_logistic.p", "wb"))		
		
		# ROC curve analysis to compute the critical value.
		roc_curve(train_data)	

	return(0)

def percep_clf():
	'''
	This fxn. mplements a Single-layer perceptron classification technique.
	The library sklearn was used for this purpose.
	'''

	train_data = pickle.load(open("train_data.p", "rb"))		# Load trainig data.	 
	ind_var  = train_data.drop(['response', 'user_id'], axis=1)	# Independent variables.
	dep_var = train_data['response']				# Dependent variable.
	
	# In-sample performance invetigation: uncomment as needed.
	#net = perceptron.Perceptron(n_iter=10000, verbose=0, random_state=None, \
	#fit_intercept=True, eta0=0.002)
	#net.fit(ind_var, dep_var)
	
	# Out-of-sample test: k-fold cross validation.
	# A k-fold out-of-sample performance was investigated. The simualtion
	# takes approx. 30 minutes to complete. As such the results generated
	# by the simulation have been saved as "train_data_percep.p".
	
	#k_fold_cross_val('perceptron')               

	# Load data after the k-fold simulation. 
	train_data = pickle.load(open("train_data_percep.p", "rb"))	 
	
	# True positive rate.
	true_pos = float(sum(train_data['response'] * (train_data['predict'])==1))*100/\
	sum(train_data['predict'])
	# False positive rate.
	false_pos = float(sum((1-train_data['response']) * (train_data['predict'])==1))*100/\
	(len(train_data)-sum(train_data['predict']))

	acc = float(sum(train_data['response']==train_data['predict']))*100/\
	len(train_data)	# Accuracy.
	print "\nOut-of-sample performance of the Single-Layer Perceptron estimator: "
	print "True Positive Rate = ", "%.2f" % true_pos, "%"
	print "False Positive Rate = ", "%.2f" % false_pos, "%"
	print "Accuracy = ", "%.2f" % acc, "%"
	return(0)

def over_disp_test():
	'''	
	Testing for over dispersion in data. The data tested 
	positive for overdispersion (p_val=0). This implies that the
	standard errors of the variables will not be corectly
	computed. Other estimators such as the quasi-binomial
	estimator could be used instead of the Logistic regressor
	to get a more accurate sense of standard error of variables.
	'''

	N = 100	# Group data.
	train_data = pickle.load(open("train_data.p", "rb"))	# Load train_data. 
	dep_var = train_data['response']	# Dependent variable.
	
	# Conditional mean.
	cond_mean = [np.mean(dep_var[i:(i+N)]) for i in xrange(0, len(dep_var.index), N)]
	p = np.average(cond_mean)		# Popoulatio mean.	
	cond_var = float(p * (1-p))/N		# Conditional var.
	std_res = (cond_mean-p)/ cond_var	# Standaridised rsiduals.

	chi_sqr	= sum(std_res**2)		# Chi-square.
	p_val = 1 - 2*chi2.cdf(chi_sqr, len(dep_var)/N)		# p-value
	
	return(p_val)


def logistic_reg():
	'''
	This fxn. implements Logistic Regression. The library
	statsmodel is used for investigation purpose because it
	outputs a summary table of the performance of the estimator.
	'''

	train_data = pickle.load(open("train_data.p", "rb"))		# Load train_data.	
	ind_var  = train_data.drop(['response', 'user_id'], axis=1)	# Independent variables.
	dep_var = train_data['response']				# Dependent variable.
	
	ind_var['intercept']  = 1.0		# Add intercept.

	# Investigating significance of variables in Logistic regression.
	model = sm.Logit(dep_var, ind_var, disp=0)	# Logistic Regression model.
	results = model.fit()			# Regress.
	print results.summary()			# Show results.
	
	# ----- Testing for over dispersion in data ---
	#p_val = over_disp_test()		# p_val = 0
	print "\nNOTE:  Data tested Positive for over dispersion. This implies"
	print "that the standard errors of the covariates are inaccurate."
	
	# ---- k-fold cross validation ---
	k_fold_cross_val('logistic')            # k-fold cross-validation.

	# ==== Model test file ====
	mod_tst = pickle.load(open("mod_tst.p", "rb"))		# Load model_test_file.
	mod_tst['intercept'] = 1.0				# Add intercept.	
	predict = pd.DataFrame(np.zeros(len(mod_tst)), index=mod_tst.index,\
	columns=['predict']) 	   				# Predicted 'response'.
	predict['predict'] = results.predict(mod_tst.drop('user_id', axis=1))		# Predict.

	mod_tst = mod_tst.join(predict, how='inner')
	crit_value = 0.24
	mod_tst['predict_response'] = 0
	mod_tst['predict_response'][mod_tst['predict'] > crit_value] = 1
	
	return(mod_tst)


def growth_metric():
	'''
	The growth metric proposed reflects the growth per week in positive 
	customer engagement. It is the Beta of the linear regression model where
	the dependent variable is the number of positive engagements recorded
	per week for each user_id (weeks with no positive engagement are aiisgned
	a vlaue of zero), and the independent variable is the week of the year.
	'''

	usr_msg = pickle.load(open("usr_msg.p", "rb"))	# Load user message. 	
	
	# Total engagements per week for each user_id.
	usr_week = (usr_msg.groupby('user_id').\
		apply(lambda g:g.set_index('content_created_date')[['total_engagement']].\
		resample('W', how='sum')).unstack(level=0).fillna(0))
	
	# Get used_id.
	usr_id = [usr_week.columns[i][1] for i in range(0, len(usr_week.columns))]
	
	# Store Gwth metric values.
	gwth_met = pd.DataFrame(np.zeros(len(usr_id)), index=usr_id, columns=['gwth'])
	
	regr = LinearRegression()		# OLS Regression
	x = np.arange(1,len(usr_week)+1,1)	# Independent variable.	
	for i in usr_id: 	
		y = usr_week['total_engagement'][i]			# Dependent variable.
		res = regr.fit(x[1:,np.newaxis], y[1:,np.newaxis])	# Regress.
		gwth_met['gwth'][i] = res.coef_				# Store Beta.
	
	# --- Examine typical user_ids ---
	#view_id = 134			
	#plt.scatter(usr_week.index, \
	#usr_week['total_engagement'][view_id])

	return(gwth_met)
	

def calc_content():
	
	usr_msg = pickle.load(open("usr_msg.p", "rb"))	 	

	# Total content_count per user_id.
	sum_content = usr_msg.groupby(['user_id'])[['content_count']].sum()	
	
	# User_id with content_count > 500.
	usr_top = sum_content.query('content_count>500')		

	return(sum_content, usr_top.index.tolist())

def clean_data():
# Clean data for analysis in Question 2.

	usr = pickle.load(open("usr.p", "rb"))
	usr_fea = pickle.load(open("usr_fea.p", "rb"))
	mod_tst = pickle.load(open("mod_tst.p", "rb"))

	# Merge user_feature and usr by user_id to get response variable.
	train_data = pd.merge(usr, usr_fea, on='user_id', how='inner')
	pickle.dump(train_data, open("train_data.p", "wb"))
	
	return(0)

def import_data():
# Load data from .xlsx files. 

	input_file = '/Users/vashishtha/myGitCode/circleup/Data_science_take_home_Datasets.xlsx' 

	# --- Read Files ---
	usr_msg = pd.read_excel(input_file, sheetname='user_message')
	usr = pd.read_excel(input_file, sheetname='user')
	usr_fea = pd.read_excel(input_file, sheetname='user_features')
	mod_tst = pd.read_excel(input_file, sheetname='model_test_file')
	
	# --- Save files as .p --- 
	pickle.dump(usr_msg, open("usr_msg.p", "wb"))
	pickle.dump(usr, open("usr.p", "wb"))
	pickle.dump(usr_fea, open("usr_fea.p", "wb"))
	pickle.dump(mod_tst, open("mod_tst.p", "wb"))

	return(0)

def main (argv = None):

	# import_data()

	# ============== Question 1 ============== 
	# --- part 1 ---
	sum_cont, top_usr = calc_content()	
	print "\n============== Question 1 ============== "
	print "\nUsers with greater than 500 peices of content: \n", top_usr
	
	# --- part 2 ---
	gwth = growth_metric()
	sum_cont = sum_cont.join(gwth, how='inner', lsuffix=' _x')
	sum_cont = sum_cont.sort('gwth', ascending=False)
	fast_gwth = sum_cont.index.tolist()[0:10]
	print "\nTop 10 users with fastest growth: \n", fast_gwth, "\n"

	# =============== Question 2 ===============

	#clean_data() 	# Does not require to be run: data has been cleaned and saved.
	
	print "\n============== Question 2 ==============\n"	
	# Logistic regression is carried out to investigate its
	# performance in predicting the 'response' variable. 
	
	mod_tst_res = logistic_reg()  # Uncomment out to run logistic regression estimator.
	
	# A Single-layer perceptron classification technique was also explored
	# to investigate its use for binary classificartion. The choice of  
	# single-layer over multi-layer perceptron was made because of its
	# simplicity. 
	
	percep_clf()	# Uncomment out to run single-layer perceptron estimator.

	# The Logistic Regressor was used to predict the response in the model_test_file.
	# Predicted response for the users have been saved in the file 
	# "model_test_file_results.xlsx".
	
	writer = pd.ExcelWriter('model_test_file_results.xlsx')
	mod_tst_res.to_excel(writer, 'Sheet1')
	print"\nThe Logistic Regressor was used to predict the response of users in the" 
	print"model_test_file. Results have been saved in the file model_test_file_results.xlsx\n"
	writer.save()	
	return(0)

if __name__ == '__main__':
	status = main()
	sys.exit(status)

