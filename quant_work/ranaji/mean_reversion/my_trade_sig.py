
from my_quant_library import *

def momt_rib(tik_sim, asset_name):
	'''
	The strategy is as follows:
	* LONG: if the Similarity Value is 0 (i.e. rolling mean n > ... > m).
	where n > m
	* SHORT: if Similarity Value is 1 (i.e. rolling mean n < ... < m) 
	where n > m
	'''

	# Get colum name for Similartity Value (sim. value).
	col_name = tik_sim.columns[0]
	
	# Column name for trading sigs. 
	asset_sig = '%s_sig' %asset_name
	
	# Column for storing trading signal.
	sig = pd.DataFrame(np.zeros(len(tik_sim)), index=tik_sim.index, \
	columns=[asset_sig])
	
	tik_sim = tik_sim.join(sig, how='inner', lsuffix=' _x')

	lev = 1		# Leverage: amount of stock to buy or sell.


	for i in range(1, len(tik_sim)):	# Iterate thorugh dates.

		# Long condition.
		if (tik_sim[col_name][i] == 0):
			tik_sim[asset_sig][i] = lev

		# Short condition.
		else:	tik_sim[asset_sig][i] = -lev
	
	return(tik_sim)


def momt_ro(tik_sim, asset_name):
	'''
	The strategy is as follows:
	* LONG: if the Similarity Value is 0 (i.e rolling mean- 30 > 60 > 90)
	AND the absolute difference between the 30- and 60- day rolling mean
	is less than the mean of daily historical difference (since trading 
	date) minus one std.dev. OR
	* LONG: if the Similarity Value is 0 AND the last 10 days into the  
	trade have had Similarity Values of 0.
	* SHORT: if the Similarity Value is 1 (i.e rolling mean- 30 < 60 < 90)
	AND the absolute difference between the 30- and 60- day rolling mean 
	is less than the mean of daily historical difference (since trading 
	date) minus one std.dev. OR
	* SHORT: if the similarity value is 1 AND the last 10 days into the 
	trade have had Similarity Value of 1.
	* NO TRADE: if none of the above are not satisfied.  
	'''

	# Get colum name for Similartity Value (sim. value).
	col_name = tik_sim.columns[0]		
	asset_sig = '%s_sig' %asset_name	# Column name for trading sigs. 
	
	# Column for storing trading signal.
	sig = pd.DataFrame(np.zeros(len(tik_sim)), index=tik_sim.index, \
		columns=[asset_sig])

	tik_sim = tik_sim.join(sig, how='inner', lsuffix=' _x')

	lev = 1		# Leverage.
	buy = 1 	# Bit to indicate if previous trade was not a Long (=1).
	sell = 1	# Bit to indicate if previous trade was not a Short (=1).

	# Column names of the smallest and largest rolling mean 
	# days (eg. 'assetnamw_30') 
	[sm_mean, lg_mean] = [i for i in tik_sim.columns if i[-1].isdigit()]

	# Indicate the date the trade was entered into.
	trade_date = tik_sim.index[0]

	# Mean - 1 std. dev. of absolute difference between 30- and 60- day 
	# rolling mean. The initial (single) value has a std. dev. of 0. 
	chk_val = abs(tik_sim[sm_mean][trade_date] - \
		tik_sim[lg_mean][trade_date]) 

	trail = 0	# No. days since entering into the trade.
	for i in tik_sim.index:		# Iterate through dates.
		
		# Cuurent absolute difference between 30- and 60- day 
		# rolling mean.
		cur_dif = abs(tik_sim[sm_mean][i] - tik_sim[lg_mean][i]) 

		# Long condition.
		if ((tik_sim.loc[i][col_name] == 0 and chk_val <= cur_dif) or \
			(tik_sim.loc[i][col_name] == 0 and trail >= 10)):

			# If the previous trade was not a Long trade. 
			if (buy == 1):		
				trade_date = i	# Date of entering the trade.
				lev = 1		# Set leverage (amount of stock)
				
				# Set to 1 so that the next trade will see that 
				# the previous trade was not Short.
				sell = 1	

				# Set to 1 so that the next trade will see that 
				# the previous trade was Long . 
				buy = 0		
			
			# Increment the no. days of since entering into the 
			# trade. 
			trail += 1	 
			
			# Set trade signal to Buy. 
			tik_sim[asset_sig][i] = lev	
			
		# Short condition.
		elif ((tik_sim.loc[i][col_name] == 1 and chk_val <= cur_dif) or \
			(tik_sim.loc[i][col_name] == 1 and trail >= 10)):
			
			# If the previous trade was not a Short trade. 
			if (sell == 1):
				trade_date = i	# Date of entering the trade.
				lev = 1		# Set levarage (amount of stock).
				
				# Set to 1 so that the next trade will see that 
				# the previous trade was not Long. 
				buy = 1	
				
				# Set to 1 so that the next trade will see that 
				# the previous trade was Short.
				sell = 0

			# Increment the no. days of since entering into the 
			# trade. 
			trail += 1
		
			# Set trade signal to Sell. 
			tik_sim[asset_sig][i] = -lev
	
		# If neither Long nor Short.
		else:
			lev = 0		# Amount of stock to trade (=0).
			
			# Set trade signal to no trade. 
			tik_sim[asset_sig][i] = lev
			
			trade_date = i	# Date of entering into the trade.
			
			# Set to 1 that the next trade will see that the 
			# previous trade was not Long. 
			buy = 1
			
			# Set to 1 that the next trade will see that the 
			# previous trade was not Short.
			sell = 1

			# Set the no. days of since entering into the trade. 
			trail = 0

		# Dates that the trade has occurred.
		past_date = tik_sim.index[((tik_sim.index>=trade_date) & \
				(tik_sim.index <= i))]
		
		# Daily absolute difference between 30- and 60- day rolling 
		# mean since entering into the trade. 
		dif_array = abs(tik_sim[sm_mean][past_date] - \
				tik_sim[lg_mean][past_date])
		
		# Mean - 1 std. dev. of daily absolute difference between 30-
		# and 60- day rolling mean since entering into the trade.
		chk_val = np.mean(dif_array) - 1*np.std(dif_array)


	return(tik_sim)
