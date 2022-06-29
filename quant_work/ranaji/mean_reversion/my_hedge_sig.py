
from my_quant_library import *

def index_ret(trade_sig):

	# Names of assets.
 	name_asset = [i for i in trade_sig.columns if '_' not in str(i)][0]
	
	# Names of columns with Trade Signals.
 	name_sig = [i for i in trade_sig.columns if '_sig' in str(i)][0]

	# Identify trade pts.
	trade_pt = - trade_sig[name_sig] + trade_sig[name_sig].shift(periods=1, \
			axis=0) 

	# Identify Last day trade price.
	trade_pt[-1] = abs(trade_sig[name_sig][-1])
	
	# Identify First day trade price.
	trade_pt[0] = abs(trade_sig[name_sig][0])
	
	asset_fut_ret = '%s_fut_ret' %name_asset 
	asset_fut_pri = '%s_fut_pri' %name_asset 
	trade_sig[asset_fut_ret] = 0	
	sum_price = 0
	k = 0	# Closing trade.
	for i in trade_sig.index[::-1]:

		if (trade_pt.ix[i] != 0):
			# Check if the trading pt. a double trading pt. 
			# (ie. one where a trade is closed and opened).         
			sum_price = trade_sig.ix[i][asset_fut_pri] + sum_price
			if (abs(trade_pt.ix[i]) != 2): 
			# If not a double trading pt. 
				if (k == 0):	# If not a trading period then 
					k = 1	# identify trading period (closing pt. of trade). 
                                        cls_date = i
				else:
				#	trade_sig.loc[i, asset_fut_ret] = \
                                #                float(sum_price)/abs(trade_sig.ix[i][asset_fut_pri])
					trade_sig.loc[cls_date, asset_fut_ret] = \
                                                float(sum_price)/abs(trade_sig.ix[i][asset_fut_pri])
					k = 0   # Identify close of trading (opening pt. of trade).
					sum_price = 0	# Reset.
                                        cls_date = 0

			else:	# A double trade.
				# Adjust the trading price to the correct price (i.e. divide by 2).

				# Compute returns.
			#	trade_sig.loc[i, asset_fut_ret] = \
			#			float(sum_price)/abs(trade_sig.ix[i][asset_fut_pri])

				trade_sig.loc[cls_date, asset_fut_ret] = \
                                                float(sum_price)/abs(trade_sig.ix[i][asset_fut_pri])
				# Since a double trading pt. keep the closing trade price!!
				sum_price = trade_sig.ix[i][asset_fut_pri]
                                cls_date = 0
				
	return()


def beta_reg(assetX, assetY, nameX, nameY):

	assetY = pd.DataFrame(assetY)	
	assetX = pd.DataFrame(assetX)	

	assetY.index = pd.to_datetime(assetY.index)
	assetX.index = pd.to_datetime(assetX.index)

	data = assetY.join(assetX, how='inner')

	data = sm.add_constant(data)
	model = sm.regression.linear_model.OLS(data[nameY], \
		data.drop([nameY], axis=1))
	results = model.fit()
	#print results.summary()

	beta = results.params[1]	
	
	return(beta)

def snp_fut(trade_sig, hedge_asset):
	
	# Names of assets.
 	name_asset = [i for i in trade_sig.columns if '_' not in str(i)][0]

	# Optimal hesge ratio.
	beta = beta_reg(hedge_asset['Price'], trade_sig[name_asset], \
		'Price', name_asset)

	# Names of columns with Trade Signals.
 	name_pri = [i for i in trade_sig.columns if '_pri' in str(i)][0]

	asset_fut_pri = '%s_fut_pri' %name_asset 
	trade_sig = trade_sig.join(hedge_asset['Price'], how='inner')

	trade_sig.rename(columns={'Price':asset_fut_pri}, inplace=True)
	trade_sig[asset_fut_pri]= -1 * np.sign(trade_sig[name_pri]) * trade_sig[asset_fut_pri]

	index_ret(trade_sig)

	# Names of assets.
 	name_fut_ret = [i for i in trade_sig.columns if '_ret' in str(i)][1]

        beta = 1
	trade_sig[name_fut_ret] = beta * trade_sig[name_fut_ret]

	return(trade_sig)

