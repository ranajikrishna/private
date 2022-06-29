
from my_quant_library import *

def get_price(tikr, start_date, end_date, prd):
	'''
		Returns a dataframe of prices of tickers between the start_date 
		and end_date. The type of price (Open, High, Low, Close) is 
		specified by prd. 
		class pandas.DataFrame(list, str, str, str).
	'''

        pdb.set_trace()
	# Get historical data from Yahoo.
	stcks = [Share(x).get_historical(start_date, end_date) for x in tikr]

        pdb.set_trace()
	# Remove tickers that were not found. 
	stcks = [j for j in stcks if len(j)!=0]    
	
        pdb.set_trace()
	# Get list of tickers that returned prices.
	pop_tikr = [x[0]['Symbol'] for x in stcks] 

        pdb.set_trace()
	# Set of tickers that did not return prices.
	miss_tikr = set(tikr) - set(pop_tikr)	   
	
	# Print tickers that did not return prices.   
	if (len(miss_tikr)!=0):
		print "Can't find any prices for the following tickers: "
		print ", ".join(x for x in miss_tikr)

	# Populate with prices for the first asset.
	data = pd.DataFrame([x[prd] for x in stcks[0]],\
	                    index = [x['Date'] for x in stcks[0]],\
		            columns = tikr[0:1])
	
	# Re-index so that dates are in ascending order. 
	data = data.reindex(index=data.index[::-1]) 		
	
        pdb.set_trace()
	# Populate with prices of the other tickers.
	for i in range(1, len(stcks)):
		a_y = 0
		new_tikr = pd.DataFrame([x[prd] for x in stcks[i]],\
	        	            index=[x['Date'] for x in stcks[i]],\
		        	    columns=tikr[i:i+1])
		
		# Merge (Union) prices with all populated tickers.
		data = data.join(new_tikr, how='outer', lsuffix=' _x')

	return(data.astype(float))

