
from my_quant_library import *


def ham_score(val_matrix, ind):

	sz = np.shape(val_matrix)[0]		# Size of vector.
	
	# Indices of each of the sorted columns of matrix.
	val_srt_list = [np.argsort(val_matrix[x]) for x in val_matrix.columns]
	
	# Columns with Nan are given indices of -1. Convert them to NaN. 
	val_srt_list = [x*float('NaN') if np.sum(x >= 0) != sz else x for x in val_srt_list]

	# Store similarity. 
	sim = pd.DataFrame(np.zeros(np.shape(ind)), index=ind, columns=['tmp'])  
	j = 1
	for i in ind[1:]:
		# Similarity btw. index[i] & index[i-1] using Hamming distance: 
		# 0 (similiar) 1 (unsimilar).	
		sim.loc[i] = sp.spatial.distance.hamming(val_srt_list[j], val_srt_list[j-1])
		j += 1

	return(sim)

def sim_score(val_matrix, ind):

	sz = np.shape(val_matrix)[0]		# Size of vector.
	ori_vec = range(sz-1, -1, -1)		# Original Vector: 
	up_bound = np.sum(np.array(ori_vec)**2)
	lo_bound = np.dot(range(sz), ori_vec)
	
	# Indices of each of the sorted columns of matrix.
	val_srt_list = [np.argsort(val_matrix[x]) for x in val_matrix.columns]
	
	# Columns with Nan are given indices of -1. Convert them to NaN. 
	val_srt_list = [x*float('NaN') if np.sum(x >= 0) != sz else x for x in val_srt_list]

	sim = pd.DataFrame(np.zeros(np.shape(ind)), index = ind)  # Store similarity. 
	j = 0
	for i in ind:
		# Similarity between val_srt_list and [n, n-1, ..., 0]^T.  
		val = float(np.dot(val_srt_list[j], ori_vec))
		# Scale val to between 0 (very similiar) and 1 (unsimilar).	
		sim.loc[i] = float(val - up_bound)/float(lo_bound - up_bound)
		j += 1

	return(sim)

