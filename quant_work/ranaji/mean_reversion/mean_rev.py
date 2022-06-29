
'''
Name: Simple mean reverting model.

Author: Ranaji Krishna.

Notes:
Here is a simple mean-reverting model that is due to Amir Khan-dani and Andrew Lo at MIT:
web.mit.edu/alo/www/Papers/august07.pdf

'''


from myLib import *


def clean_data(stocks):					#  Clean data

	all_stocks = pd.HDFStore('all_stocks.h5')	# Load HDFS data.
	uni = all_stocks['all_stocks']['Open']		# Get Open data.
	uni = uni.dropna(axis=1, how = 'all')		# Drop tickers with all nan.
       
        # --- Clean Manually in Excel ---
	writer = pd.ExcelWriter('output.xls')		
	uni.to_excel(writer,'Sheet1')
	writer.save()					# Save to Excel for manual cleaning.	
	
	inputFile = '/Users/vashishtha/myGitCode/quant/output_clean.xls'   # Load from Excel.

	uni = pd.ExcelFile(inputFile).parse().dropna()	# Drop rows with nans.

	return(uni)

def load_data(inputFile):		# Load data

	tik = pd.ExcelFile(inputFile)	# Ticker names.
	#outputFile = 'SPX_20071123'
	
	try:	
		# Load data from Yahoo! Finance. 
		all_stocks = DataReader(tik.parse('21-NOV-2007_500')['Symbol'],  'yahoo', datetime(2000,1,1), datetime(2007,11,23))

		store = pd.HDFStore('all_stocks.h5')	# Save as HDFS.
		store['all_stocks'] = all_stocks
	except:
		print "Can't find ", tik.parse('21-NOV-2007_500')['Symbol']

	return(all_stocks)	

def main(argv = None):

	inputFile = '/Users/vashishtha/myGitCode/quant/SP500_20071121.xls'      # Input filename.

	#all_stocks = load_data(inputFile)	# Load data.
	#universe = clean_data(all_stocks)	# Clean data.
	#universe.to_pickle('/Users/vashishtha/myGitCode/myProj/myCodePy/quant/universe.pkl')   # Save as a pickle object.

	# Load clean universe (pickle object).
	universe = pd.read_pickle('/Users/vashishtha/myGitCode/quant/universe.pkl')
	dailyret = (universe - universe.shift(periods=1))/universe.shift(periods=1)		# Daily returns. 	
	mean_ret = dailyret.mean(axis = 1)	# Mean of daily returns.

	# Weights. Note: Weights should add to 0 since dollar neutral long-short strategy. 
	weights = -(dailyret - mean_ret)/np.shape(dailyret)[1]		
	
	# Daily profit and loss.
	dailypnl = (weights.shift(1) * dailyret).sum(axis=1)

	onewaycost = (5/100) * 1/100	# One way transaction cost.
	
	# Daily profit and loss with transaction cost accounted for.
	dailypnl = dailypnl - abs(weights - weights.shift(1)).sum(axis=1) * onewaycost	
	
	sharpe = sqrt(252) * dailypnl.mean() / sqrt(dailypnl.var())	# Sharpe ratio.

	return(0)


if __name__ == '__main__':
	status = main()
	sys.exit(status)
