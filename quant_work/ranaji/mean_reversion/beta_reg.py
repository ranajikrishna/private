
from my_quant_library import *

def beta_reg(assetX, assetY):
	
	pdb.set_trace()
	assetY.index = pd.to_datetime(assetY.index)
	assetX.index = pd.to_datetime(assetX.index)

	data = assetY.join(assetX, how='inner')

	assetY = assetY[assetX.index]
	assetY = assetY.dropna()
	assetX = assetX[assetY.index]
	assetX = assetX.dropna()
	data = sm.add_constant(data)
	model = sm.regression.linear_model.OLS(assetY, assetX)
	results = model.fit()
	#print results.summary()

	beta = results.params[1]	
	
	return(beta)

#def main():
#	
#	# Input filename.
#	inputFile = '/Users/vashishtha/myGitCode/quant/ranaji/snp500_futures.xlsx' 
#	snp_fut = pd.read_excel(inputFile, sheetname='Sheet1', index='Date')	# SNP Futures data.
#	snp_fut.set_index('Date',inplace=True)
#	
#	universe = pd.HDFStore('mom_data.h5')	# Load HDFS data.
#	beta_reg(snp_fut['Price'], universe['data'].AAPL['2013-01-02':'2015-01-12'])
#	
#	return(0)


#if __name__ == '__main__':
#	status = main()
#	sys.exit(status)
		
	
	
