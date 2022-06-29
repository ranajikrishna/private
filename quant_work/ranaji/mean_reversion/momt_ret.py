
from my_quant_library import *


def cml_ret (trade_sig, ret_typ):
    
        asset_pri = [i for i in trade_sig.columns if '_' not in str(i)]
        asset_ret = [i+ret_typ for i in trade_sig.columns if '_' not in str(i)]

        tmp_ret = trade_sig[asset_ret]
        ret = trade_sig[asset_ret]
        pri = trade_sig[asset_pri]

        tmp_ret[tmp_ret != 0] = 1
        wgt = pd.DataFrame(tmp_ret.shift(-1).values*pri.values, columns=pri.columns, index=pri.index)
        wgt = pd.DataFrame([wgt[i]/np.sum(wgt,1) for i in asset_pri])
        wgt_ret = pd.DataFrame(ret.shift(-1).values*(wgt.values).T, columns=pri.columns, index=pri.index)
        
        return(np.sum(wgt_ret,1))

def comp_ret(trade_sig):

	# Names of columns with Trade Signals.
 	name_sig = [i for i in trade_sig.columns if '_sig' in str(i)][0]

	# Names of assets.
 	name_asset = [i for i in trade_sig.columns if '_' not in str(i)][0]

	# Identify trade pts.
	trade_pt = - trade_sig[name_sig] + trade_sig[name_sig].shift(periods=1, \
			axis=0) 

	# Identify trade prices.
	trade_sig['%s_pri' %name_asset] = trade_sig[name_asset] * trade_pt

	# Identify Last day trade price.
	trade_sig['%s_pri' %name_asset][-1] = trade_sig[name_asset][-1] * \
						trade_sig[name_sig][-1]
	
	# Identify First day trade price.
	trade_sig['%s_pri' %name_asset][0] = trade_sig[name_asset][0] * \
						-1 * trade_sig[name_sig][0]
	
	# Names of columns with Trade Signals.
 	name_pri = [i for i in trade_sig.columns if '_pri' in str(i)][0]

	# Calculate price.
	trade_sig['%s_ret' %name_asset] = 0
	
	# Names of columns with Trade Signals.
 	name_ret = [i for i in trade_sig.columns if '_ret' in str(i)][0]

	# Compute returns.
	sum_price = 0
	k = 0	# Closing trade.
        pre_ret = 0
	for i in trade_sig.index:
	        
		if (trade_sig.ix[i][name_pri] != 0):	# Trading point.
			# Check if the trading pt. a double trading pt. 
			# (ie. one where a trade is closed and opened).           
			if (abs(trade_sig.ix[i][name_pri]) == trade_sig.ix[i][name_asset]): 
			# If not a double trading pt.	
				if (k == 1):	# If not a trading period then 
                                        #trade_sig.loc[i, name_ret] = ((float(trade_sig.ix[i][name_asset]) /abs(pre_price) - 1) * np.sign(pre_price) + 1) * (1 + pre_ret) - 1 
                                        trade_sig.loc[i, name_ret] = (float(trade_sig.ix[i][name_asset])/abs(pre_price) - 1) * np.sign(pre_price)
                                        pre_ret = 0
                                        k = 0
                                else:
                                        pre_price = trade_sig.ix[i][name_asset] * trade_sig.ix[i][name_sig] 
                                        k = 1

			else:	# A double trade.
				# Adjust the trading price to the correct price (i.e. divide by 2).
				trade_sig.ix[i][name_pri] = float(trade_sig.ix[i][name_pri])/2

				if (k == 1):	# If not a trading period then 
                                        #trade_sig.loc[i, name_ret] = ((float(trade_sig.ix[i][name_asset]) /abs(pre_price) - 1) * np.sign(pre_price) + 1) * (1 + pre_ret) - 1 
                                        trade_sig.loc[i, name_ret] = (float(trade_sig.ix[i][name_asset])/abs(pre_price) - 1) * np.sign(pre_price)
                                        pre_ret = 0
                                else:
                                        pre_price = trade_sig.ix[i][name_asset] * trade_sig.ix[i][name_sig] 
                                        k = 1

                elif (k == 1):
                            #trade_sig.loc[i, name_ret] = ((float(trade_sig.ix[i][name_asset]) /abs(pre_price) - 1) * np.sign(pre_price) + 1) * (1 + pre_ret) - 1 
                            trade_sig.loc[i, name_ret] = (float(trade_sig.ix[i][name_asset])/abs(pre_price) - 1) * np.sign(pre_price)
                            pre_ret = trade_sig.ix[i][name_ret]
                            pre_price = trade_sig.ix[i][name_asset] * trade_sig.ix[i][name_sig] 
                
	return(trade_sig)

def comp_ret1(trade_sig):

	# Names of columns with Trade Signals.
 	name_sig = [i for i in trade_sig.columns if '_sig' in str(i)][0]

	# Names of assets.
 	name_asset = [i for i in trade_sig.columns if '_' not in str(i)][0]

	# Identify trade pts.
	trade_pt = - trade_sig[name_sig] + trade_sig[name_sig].shift(periods=1, \
			axis=0) 

	# Identify trade prices.
	trade_sig['%s_pri' %name_asset] = trade_sig[name_asset] * trade_pt

	# Identify Last day trade price.
	trade_sig['%s_pri' %name_asset][-1] = trade_sig[name_asset][-1] * \
						trade_sig[name_sig][-1]
	
	# Identify First day trade price.
	trade_sig['%s_pri' %name_asset][0] = trade_sig[name_asset][0] * \
						-1 * trade_sig[name_sig][0]
	
	# Names of columns with Trade Signals.
 	name_pri = [i for i in trade_sig.columns if '_pri' in str(i)][0]

	# Calculate price.
	trade_sig['%s_ret' %name_asset] = 0
	
	# Names of columns with Trade Signals.
 	name_ret = [i for i in trade_sig.columns if '_ret' in str(i)][0]

	# Compute returns.
	sum_price = 0
	k = 0	# Closing trade.
	for i in trade_sig.index[::-1]:
		
		if (trade_sig.ix[i][name_pri] != 0):	# Trading point.
			# Check if the trading pt. a double trading pt. 
			# (ie. one where a trade is closed and opened).           
			if (abs(trade_sig.ix[i][name_pri]) == trade_sig.ix[i][name_asset]): 
			# If not a double trading pt.	
				sum_price = trade_sig.ix[i][name_pri] + sum_price
				if (k == 0):	# If not a trading period then 
					k = 1	# identify trading period (closing pt. of trade). 
                                        cls_date = i

				else:
				#	trade_sig.loc[i, name_ret] = \
				#		float(sum_price)/abs(trade_sig.ix[i][name_pri])
					trade_sig.loc[cls_date, name_ret] = \
						float(sum_price)/abs(trade_sig.ix[i][name_pri])
					k = 0   # Identify close of trading (opening pt. of trade).
					sum_price = 0	# Reset.
                                        cls_date = 0

			else:	# A double trade.
				# Adjust the trading price to the correct price (i.e. divide by 2).
				trade_sig.ix[i][name_pri] = float(trade_sig.ix[i][name_pri])/2

				# Compute returns.
				sum_price = float(trade_sig.ix[i][name_pri])/2 + sum_price 
			#	trade_sig.loc[i, name_ret] = \
			#			float(sum_price)/abs(trade_sig.ix[i][name_pri])

				trade_sig.loc[cls_date, name_ret] = \
					float(sum_price)/abs(trade_sig.ix[i][name_pri])
				# Since a double trading pt. keep the closing trade price!!
				sum_price = trade_sig.ix[i][name_pri]
                                cls_date = 0

	return(trade_sig)
		

def comp_ret2(trade_sig):

	# Names of columns with Trade Signals.
 	name_sig = [i for i in trade_sig.columns if '_sig' in str(i)][0]

	# Names of assets.
 	name_asset = [i for i in trade_sig.columns if '_' not in str(i)][0]

	# Identify trade pts.
	trade_pt = - trade_sig[name_sig] + trade_sig[name_sig].shift(periods=1, \
			axis=0) 

	# Identify trade prices.
	trade_sig['%s_pri' %name_asset] = trade_sig[name_asset] * trade_pt

	# Identify Last day trade price.
	trade_sig['%s_pri' %name_asset][-1] = trade_sig[name_asset][-1] * \
						trade_sig[name_sig][-1]
	
	# Identify First day trade price.
	trade_sig['%s_pri' %name_asset][0] = trade_sig[name_asset][0] * \
						-1 * trade_sig[name_sig][0]
	
	# Names of columns with Trade Signals.
 	name_pri = [i for i in trade_sig.columns if '_pri' in str(i)][0]

	# Calculate price.
	trade_sig['%s_ret' %name_asset] = 0
	
	# Names of columns with Trade Signals.
 	name_ret = [i for i in trade_sig.columns if '_ret' in str(i)][0]

	# Compute returns.
	sum_price = 0
	k = 0	# Closing trade.
	for i in trade_sig.index[::-1]:
		
		if (trade_sig.ix[i][name_pri] != 0):	# Trading point.
			# Check if the trading pt. a double trading pt. 
			# (ie. one where a trade is closed and opened).           
			if (abs(trade_sig.ix[i][name_pri]) == trade_sig.ix[i][name_asset]): 
			# If not a double trading pt.	
				sum_price = trade_sig.ix[i][name_pri] + sum_price
				if (k == 0):	# If not a trading period then 
					k = 1	# identify trading period (closing pt. of trade). 
				else:
					trade_sig.loc[i, name_ret] = \
						float(sum_price)/abs(trade_sig.ix[i][name_pri])
					k = 0   # Identify close of trading (opening pt. of trade).
					sum_price = 0	# Reset.

			else:	# A double trade.
				# Adjust the trading price to the correct price (i.e. divide by 2).
				trade_sig.ix[i][name_pri] = float(trade_sig.ix[i][name_pri])/2

				# Compute returns.
				sum_price = float(trade_sig.ix[i][name_pri])/2 + sum_price 
				trade_sig.loc[i, name_ret] = \
						float(sum_price)/abs(trade_sig.ix[i][name_pri])

				# Since a double trading pt. keep the closing trade price!!
				sum_price = trade_sig.ix[i][name_pri]

	return(trade_sig)
