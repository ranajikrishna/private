
from my_quant_library import *

def tradeSig(data, trade_sig, fut_price):

	# ======= Plot prices and ribbons =======

	# Names of assets.
 	name_asset = [i for i in trade_sig.columns if '_' not in str(i)]
	name_sig = ['%s_sig' %i for i in name_asset]	

	m = 5; p = 91; q = 5;
	#interval = range(m, p, q)
	interval = [30, 60, (p-1)]			# N-day rolling means.  
	with PdfPages('mom_mov_av.pdf') as pdf:		# Print to pdf.
		for i in name_asset:			# Iterate through assets.
			
			# Select assets.
			asset = data[i]			 
			fig = asset.plot().get_figure()

			# Store rolling means.
			#ro_mean = pd.DataFrame(np.zeros((len(interval), \
			#np.shape(data)[0])), index=interval)
			
			ro_mean = pd.DataFrame(np.zeros((np.shape(data)[0],\
				len(interval))), index=data.index, columns=interval)
			
			# Iterate through rolling mean day.
			for j in interval:	
				# j-day rolling mean.
				ro_mean[j] = pd.rolling_mean(asset, j)  
				
				# --- Plot rolling-means ----
				fig = ro_mean[j].plot(label=str(j)).get_figure()
				plt.ylabel('Price')
				locs, labels = plt.xticks()
			
			# ---- Plot trade signals ----
			name_sig = '%s_sig' %i		# assetname_sig; eg 'AAPL_sig'
			# Position the trade signal at the mean price and scale +1/-1
			# to +2/-2.
			ro_mean = ro_mean.join(np.mean(trade_sig[i]) + \
				2*trade_sig[name_sig], how='left')
			
			# ---- Plot points of trade ---- 
			name_pri = '%s_pri' %i		# assetname_pri; eg 'AAPL_pri'.

			# Points of Shorting.
			short_pt = trade_sig[trade_sig[name_pri]>0][name_pri]
			ro_mean = ro_mean.join(short_pt, how='left')

			# Rename column.
			ro_mean = ro_mean.rename(columns={name_pri:'short_pt'})	
			
			# Points of going Long. Note: Multiple by -1 to make +ve.
			long_pt = abs(trade_sig[trade_sig[name_pri]<0][name_pri])
			ro_mean = ro_mean.join(long_pt, how='left')
			# Rename column.
			ro_mean = ro_mean.rename(columns={name_pri:'long_pt'})	
			
			# --- Plot ---
			fig = ro_mean[name_sig].plot(label=name_sig).get_figure()	# Trade signal.
			fig = ro_mean['short_pt'].plot(label='short', marker='o').get_figure()	# Short pts. 
			fig = ro_mean['long_pt'].plot(label='long', marker='s').get_figure()	# Long pts.
	
			plt.grid()	
			plt.setp(labels, rotation=45)
			datacursor(hover=True)		# Set data cursor.
			plt.legend(loc='lower right')
			pdf.savefig(fig)			
			#fig.clear()			# Must clear fig!!!!!
			plt.close(fig)

	with PdfPages('fut_sig.pdf') as pdf:		# Print to pdf.
		for i in name_asset:			# Iterate through assets.
			
			tmp_df = fut_price

			fig1 = tmp_df.plot(label='snp_fut').get_figure()
			
			# ---- Plot points of trade ---- 
			name_fut_pri = '%s_fut_pri' %i		# assetname_pri; eg 'AAPL_pri'.
			name_short = '%s_short' %i
			name_long = '%s_long' %i
			name_sig = '%s_sig' %i		# assetname_sig; eg 'AAPL_sig'

			# Points of Shorting.
			short_pt = trade_sig[trade_sig[name_fut_pri]>0][name_fut_pri]
			tmp_df = tmp_df.join(short_pt, how='left')
			tmp_df.rename(columns={name_fut_pri: name_short}, inplace=True)

			# Points of going Long. Note: Multiple by -1 to make +ve.
			long_pt = abs(trade_sig[trade_sig[name_fut_pri]<0][name_fut_pri])
			tmp_df = tmp_df.join(long_pt, how='left')
			tmp_df.rename(columns={name_fut_pri: name_long}, inplace=True)
			

			tmp_df = tmp_df.join(np.mean(tmp_df['Price']) + \
				100*trade_sig[name_sig], how='left')

			fig1 = tmp_df[name_sig].plot(label=name_sig).get_figure()	# Short pts.
			fig1 = tmp_df[name_long].plot(label='long', marker='s').get_figure()		# Long pts.
			fig1 = tmp_df[name_short].plot(label='short', marker='o').get_figure()	# Short pts.

			plt.grid()	
			plt.setp(labels, rotation=45)
			datacursor(hover=True)		# Set data cursor.
			plt.legend(loc='lower right')
			pdf.savefig(fig1)			
			#fig1.clear()			# Must clear fig!!!!
			plt.gcf().clear()	
			del tmp_df
			#plt.close('all')
	return(0)



