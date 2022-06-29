
from my_quant_library import *


def max_drawdown(itr_sig):

	# Names of assets.
 	name_asset = [i for i in itr_sig.columns if '_' not in str(i)][0]
	
	# Names of columns with Trade Signals.
 	name_sig = [i for i in itr_sig.columns if '_sig' in str(i)][0]

	# Names of columns with Trade Signals.
 	name_pri = [i for i in itr_sig.columns if '_pri' in str(i)][0]

	# Identify itr pts.
	trade_pt = - itr_sig[name_sig] + itr_sig[name_sig].shift(periods=1, \
			axis=0) 

	# Identify Last day trade price.
	trade_pt[-1] = abs(itr_sig[name_sig][-1])
	
	# Identify First day trade price.
	trade_pt[0] = abs(itr_sig[name_sig][0])

	# Column name for trading sigs. 
	asset_drw = '%s_drw_ret' %name_asset

	# Store maximum drawdown per trade
	max_drd_ret = pd.DataFrame(np.zeros(len(itr_sig)), index=itr_sig.index, \
	columns=[asset_drw])

	k = 0	# Closing trade.
	for i in trade_pt.index[::-1]:
		
		if (trade_pt.ix[i] != 0):	# Trade pt.	

			if (k == 0):	# If not a trading period then
				close_pt = i
				# identify closing pt. of trade). 
				k = trade_pt.ix[i]	
			else:
				open_pt = i
				if (k > 0): 	# Close Trade pt. =  Sell, i.e Long Trade. 
					max_pt=itr_sig[open_pt:close_pt][name_asset].idxmax()
					min_pt=itr_sig[max_pt:close_pt][name_asset].idxmin()	
					max_drd_ret.loc[i] = 1-itr_sig.ix[max_pt][name_asset]/ \
						itr_sig.ix[min_pt][name_asset]

				else:		# Close Trade pt. =  Buy, i.e Short Trade.
					min_pt=itr_sig[open_pt:close_pt][name_asset].idxmin()	
					max_pt=itr_sig[min_pt:close_pt][name_asset].idxmax()
					max_drd_ret.loc[i] = 1-itr_sig.ix[max_pt][name_asset]/ \
						itr_sig.ix[min_pt][name_asset]
				
				k = 0   # Identify close of trading (opening pt. of trade).

				# Check if the trading pt. a double trading pt. 
				# (ie. one where a trade is closed and opened).           
				if (abs(itr_sig.ix[i][name_pri]) != itr_sig.ix[i][name_asset]): 
					
					close_pt = i
					# identify closing pt. of trade). 
					k = trade_pt.ix[i]	

	itr_sig = itr_sig.join(max_drd_ret, how='left')
	return(itr_sig)
