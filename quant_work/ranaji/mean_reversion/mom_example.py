
'''
Name: Simple Momentum model.

Author: Ranaji Krishna.

Notes:

'''

from my_quant_library import *

def load_data(assets, start, end, price_typ):
	
        pdb.set_trace()
	# --- Save data as HDF5 ---	
	data = mgp.get_price(assets, start, end, price_typ)	# Get prices.
	#store_data = pd.HDFStore('mom_data.h5')		# Save as HDFS.
	store_data['data'] = data
	universe = store_data
	#universe = pd.HDFStore('mom_data.h5')	# Load HDFS data.

	# --- SNP500 Futures ---
	#inputFile = '/Users/vashishtha/myGitCode/quant/ranaji/snp500_futures.xlsx' 
	#inputFile = '/Users/ranajikrishna/invoice2go/git_code/ranaji_q/quant/ranaji/snp500_futures.xlsx'
	#snp_fut = pd.read_excel(inputFile, sheetname='Sheet1', index='Date')	
	#snp_fut.set_index('Date', inplace=True)
	#universe['fut_data'] = snp_fut		# Save as HDFS.
	

        pdb.set_trace()
	return(universe['data'])

def main(argv = None):

	# Input filename.
	#inputFile = '/Users/vashishtha/myGitCode/quant/ranaji/SP500_20071121.xls'      
	#tik = pd.ExcelFile(inputFile)	# Ticker names.
	#assets = tik.parse('21-NOV-2007_500')['Symbol']
	
	assets = sorted(['STX', 'WDC', 'CBI', 'JEC', 'VMC', 'PG', \
			'AAPL', 'PEP', 'AON', 'DAL', 'IBM'])
	start = '2013-01-01'
	end = '2016-01-01'
	price_typ = 'Adj_Close'
 	
	data = load_data(assets, start, end, price_typ)		# Get prices.
	data.index = pd.to_datetime(data.index).date
	
	# SNP Futures data.
	#universe = pd.HDFStore('mom_data.h5')	# Load HDFS data.
	snp_fut = universe['fut_data']
	# Plot the prices just for fun
	#data.plot(figsize=(15, 7), color = ['r', 'g', 'b', 'k', 'c', 'm', 'orange',
        #                   'chartreuse', 'slateblue', 'silver'])
	#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	#plt.ylabel('Price')
	#plt.xlabel('Time');
	#plt.show()


	# ============ Test strategy =============
	m = 5; p = 91; q = 0;
	interval = [30, 60, 90]		# N-day rolling mean.
	itr_sim_ham = pd.DataFrame(index=data.index) 	# Store Hamming distance.
	itr_sim = pd.DataFrame(index=data.index) 	# Store Similarity value.
	trade_sig = pd.DataFrame(index=data.index) 	# Store trade signal.
	#trade_sig.index = pd.to_datetime(trade_sig.index).date

	writer_ret = pd.ExcelWriter('returns.xlsx')
	for i in range(np.shape(data)[1]):	# Iterate through assets.

		# Store rolling means for all n-day rolling mean.
		ro_mean = pd.DataFrame(np.zeros((len(interval), \
		np.shape(data)[0])), index=interval, columns=[data.index])
		
		asset = data.iloc[:, i]			# Select assets.
		
		# Iterate through rolling mean day.
		for j in interval:		
			# j-day average.
			ro_mean.xs(j)[:] = pd.rolling_mean(asset, j)  
		
		# --- Hamming score ---
		# Populate hamming scores.
		#val = mss.ham_score(ro_mean, data.index)	
		#itr_sim_ham['%s_sim' %assets[i]] = val

		# --- Compute Similarity Score ---
		# Populate Similarity scores.
		val = mss.sim_score(ro_mean, data.index)
		itr_sim['%s_sim' %assets[i]] = val	# Column name: assetname_sim.

		# Join Asset prices.
		itr_sim = itr_sim.join(data[assets[i]], how='inner')
		
		# Smallest rolling-day (eg. assetname_30). 
		itr_sim['%s_%s' %(assets[i], ro_mean.index[0])] = ro_mean.iloc[0]
		
		# Largest rolling-mean (eg. assetname_90). 
		itr_sim['%s_%s' %(assets[i], ro_mean.index[1])] = ro_mean.iloc[1]
	
		# --- Build Trade Signal ---
		#itr_sig = mts.momt_rib(itr_sim[p-q-1:], assets[i])
		itr_sig = mts.momt_ro(itr_sim[p-q-1:], assets[i])
		# ------
	        
	        itr_sig = momt_ret.comp_ret(itr_sig)	# Compute Momentum Returns.
                itr_sig = itr_sig.dropna()

		# --- Build Hedge Signal --- 
		# Input filename.
		snp_fut.index = pd.to_datetime(snp_fut.index).date
	        itr_sig = mhs.snp_fut (itr_sig, snp_fut)
	
		# --- Compute Maximum Drawdown ---
	 	itr_sig = mdd.max_drawdown(itr_sig)	

		# Join Trade Signal and Momentum returns to Similarity Values.
		trade_sig = trade_sig.join(itr_sig, how='inner')
                    
		# Clear dataframe fot next asset.
		itr_sim = pd.DataFrame(index=data.index) 

		# Asset returns.
		name_ret = [i for i in itr_sig.columns if '_ret' in str(i)][0:3]
		net_return = trade_sig.ix[trade_sig[name_ret[0]]!=0][name_ret]
		net_return.to_excel(writer_ret, name_ret[0])

        cml_ret = momt_ret.cml_ret(trade_sig)
	trade_sig = trade_sig.join(pd.DataFrame(cml_ret), how='inner')
        trade_sig = trade_sig.rename(columns = {0:'cml_ret'})
	writer_ret.save()
	my_plot.tradeSig(data, trade_sig, pd.DataFrame(snp_fut['Price']))		

	#print [sum(trade_sig[i][0:]) for i in trade_sig.columns if '_ret' in i]
	#print np.sum([sum(trade_sig[i][0:]) for i in trade_sig.columns if '_ret' in i])
	
	writer = pd.ExcelWriter('trade_sig_3.xlsx')
	trade_sig.to_excel(writer, 'Sheet1')
	writer.save()
	
	#writer = pd.ExcelWriter('similarity_values1.xlsx')
	#itr_sim.to_excel(writer, 'Sheet1')
	#itr_sim_ham.to_excel(writer, 'Sheet2')
	#writer.save()

	return(0)



if __name__ == '__main__':
	status = main()
	sys.exit(status)
