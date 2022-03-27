import sys
import pdb
import pandas as pd
import numpy as np
import string
from itertools import islice, product
from nltk.tokenize import word_tokenize
from datasketch import MinHash, MinHashLSH
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool, freeze_support, cpu_count
import time

def plot3D(rocs, roc_aucs):
    '''
    To plot the performance of character-level shingle. The x and y axes are
    character size and window size, and the z axis is area under the curve.

    Args: 
        rocs: tpr, fpr and threshold for all combinations of window and character
        sizes.
        roc_auc: the auc for all combinations are window and character sizes.

    '''
    l1 = [2,4,6,8]
    l2 = [3,5,7,9,11,13,15,17,19]
    iterate = list(product(np.linspace(0,len(l1)-1,len(l1)).astype(int),\
                                np.linspace(0,len(l2)-1,len(l2)).astype(int)))
    ax = plt.figure(3)
    ax = plt.axes(projection='3d')
    X,Y = np.meshgrid(l1,l2)
    Z = np.zeros(shape=(len(l2),(len(l1))))
    keys = np.array((X,Y)).T
    for i in iterate:
        Z[i[1],i[0]] = roc_aucs[keys[i[0],i[1]][0],keys[i[0],i[1]][1]]
        
    # 3D plot of AUC for combinations of window and character sizes
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_ylabel('Window size')
    ax.set_xlabel('Character size')
    ax.set_title('AUC for combination of window and character sizes')
    ax.set_zlabel('AUC');
    return 

def cnf_mat(test):
    '''
    Establishes threshold for jaccard distance and computes the confusion matrix.
    Args:
        test: dataframe of test data for similarity to be computed.
    '''
    # Load ROC data 
    with open('./data/rocs_v7.pk','rb') as f:
        rocs = pickle.load(f)
   
    # Dictionary to store AUC of ROC for all variants of window sizes or 
    # combinations of window or character sizes.
    roc_aucs = {} 
    base_fpr = np.linspace(0, 1, 101)   # False positive rates for interpolation 
    f1 = plt.figure(1) 
    for i,N in enumerate(rocs.keys()):
        tprs = [] # Store true positive rates
        # Interpolate true positive rates
        tprs.append(interp(base_fpr,rocs[N][N[1]].fpr,rocs[N][N[1]].tpr))
        # Compute AUC
        roc_aucs[N] = (auc(rocs[N][N[1]].fpr,rocs[N][N[1]].tpr))
        if N == (2,19):
            # Plot ROC curves for all variations of window sizes
            plt.plot(base_fpr,tprs[0],label="Size = " + str(N) + ", AUC = " + \
                                            str(round(roc_aucs[N],4)),alpha=0.8)
            plt.grid('on')
    plt.title('Receiver Operating Curves for window size = 19 and character size = 2')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()

    # AUC for various window sizes
#    f2 = plt.figure(2)
#    plt.plot(range(1,len(roc_aucs)+1),roc_aucs.values(),label="auc")
#    plt.title("Plot of window size against performance (AUC)")
#    plt.xlabel('Window size')
#    plt.ylabel('AUC')
#    plt.grid('on')

    # Plot auc for all combinations of window and character sizes.
    plot3D(rocs, roc_aucs)

    # ====== Performance on "test" data ========
    threshold = 0.171875
    # Compute match in test data.
    test['pred'] = 1-(test.sim<threshold).astype(int)
    # Compute confusion matrix.
    print(confusion_matrix(test.match,test.pred))
    return 

def roc_analysis(train_df,test_df,rocs,N):
    '''
    Computes the rates for ROC curve.
    Args:
        train_df: datafrmae containing train data
        test_df: datafrmae containing test data
        rocs: dictionary of ROC rates
        N: combination. Serves as key for the `rocs` dictionary.
    Return:
        rocs: dictionary of ROC rates.
    '''
    # Compute ROC
    fpr, tpr, thr = metrics.roc_curve(train_df.match,train_df.sim,pos_label=1)
    roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr,'thr': thr})
    rocs[N] = roc
    #pickle.dump(rocs,open('rocs_v3.pk',"wb"))
    return rocs


def compute_sim(df):
    '''
    Computes MinHash and the Jaccard distance, which serves as the similarity
    score.
    Args:
        df: dataframe with columns containing `descriptions` from which MinHash
            are computed. 
    Return:
        df: dataframe containing similarity scores.

    '''
    num_perm = 128
    # Compute MinHash
    m = MinHash(num_perm=num_perm)  
    for col in ['x_des_proc', 'y_des_proc']:
        df[col[0:5]+'_hash'] = df[col].apply(lambda x: m.bulk(x))

    # Compute Jaccard
    df['sim'] = df.apply(lambda i: i.x_des_hash[0].jaccard(i.y_des_hash[0]),\
                                                                        axis=1)
    return df


def similarity_classifier(df, num_perm):
    '''
    Parallel processing to compute MinHash and Jaccard distances. The dataframe
    is split over `n_cores` and processed in a parallel pool.

    Args:
        df: dataframe with `descriptions` from which MinHash are computed.
        num_perm: Number of random permutations for MinHash computations

    Return:
        df : dataframe containing similarity scores.
    '''
    n_cores = 4
    df_split = np.array_split(df,n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(compute_sim, df_split))    
    pool.close()
    pool.join()
    return df 

'''
NOTE: To try: shingles and all succeeding subset in sliding window
'''
def slide_tkn(array, size):
    '''
    Return tokens within a sliding window.
    '''
    ret_list = []
    if len(array)-size+1 <= 0: return [array]
    for i in range(len(array)-size+1):
        ret_list.append(array[i:i+size])
    return ret_list


def remove_punctuation(df, col):
    '''
    Removes punctuations from `descriptions`

    Args:
        df: dataframe with column from which punctuations are removed.
        col: column from which punctuations are removed.

    Return:
        df: dataframe with column with punctuations.
    '''
    # Types of punctuations to be removed
    punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~|'   # ∆ is not present here
    transtab = str.maketrans(dict.fromkeys(punct, ''))
    df[col] = pd.DataFrame('∆'.join(df[col].tolist()).translate(transtab).\
                                            split('∆'),columns=[col])
    return df

def slide_shingle(array, size):
    '''
    Create subsets with singles.

    Args:
        array: contains data to be fragmented into subsets of singles.
        size: Size of the sliding window containing subsets.

    Return:
        df: dataframe containing subset of singles.
        
    '''

    ret_list = []
    for i in range(len(array)-size+1):
        ret_list.append(array[i:i+size])
    return ret_list

def charactizer(df,S,N):
    '''
    Technique where `descriptions` are fragmented into character based shingles.

    Args: 
        df: dataframe with `description` from which singles are formed
        S: Size of the sliding window.
        N: Size of character fragments.
    
    Return:
        df: dataframe containing character based singles.
         
    '''

    for col in ['x_description', 'y_description']:
        # Convert the entries to lower case
        df[col] = df[col].str.lower()
        # Remove punctuation
        df = remove_punctuation(df, col)
        # Remove spaces between words.
        df[col[0:5]+'_chr'] = df[col].apply(lambda x: x.replace(' ',''))
        # Single word tokenization
        df[col[0:5]+'_chr'] = df[col[0:5]+'_chr'].apply(lambda x: 
                                                            slide_shingle(x,S))
        # Encode string to 'utf-8'
        df[col[0:5]+'_chr'] = df[col[0:5]+'_chr'].apply(lambda x: \
                                    [i.encode('utf-8','replace') for i in x])
        # Sliding window of words
        df[col[0:5]+'_proc']  =df[col[0:5]+'_chr'].apply(lambda x: slide_tkn(x,N)) 
    return df 




def tokenizer(df, N):
    '''
    Computes N-gram, word level, shingles (tokenks) from "descriptions".

    Args: 
        df: dataframe containing the column of descriptions.
        col: length of subsequence of shingles (N-gram)
    
    Returns:
        Dataframe with the column (col) converted to shingles.

    '''

    for col in ['x_description', 'y_description']:
        # Convert the entries to lower case
        df[col] = df[col].str.lower()
        # Remove punctuation
        df = remove_punctuation(df, col)
        # Word tokenization
        df[col[0:5]+'_tkn'] = df[col].apply(lambda x: word_tokenize(x))
        # Encode string to 'utf-8'
        df[col[0:5]+'_tkn'] = df[col[0:5]+'_tkn'].apply(lambda x: \
                                    [i.encode('utf-8','replace') for i in x])
        # Sliding window of words
        df[col[0:5]+'_proc']  =df[col[0:5]+'_tkn'].apply(lambda x: slide_tkn(x,N)) 
    return df

def get_match(df):
    '''
    Computes match using `id`
    '''
    df['match'] = (df.x_id == df.y_id).astype(int)
    return df

def get_data():
    # Read CSV files, fill cells with empty values with NaN and drop Unnamed 
    # columns.
    train = pd.read_csv('code_challenge_train.csv', \
                                       na_values=' ').drop('Unnamed: 0',axis=1)
    test = pd.read_csv('code_challenge_test.csv', \
                                       na_values=' ').drop('Unnamed: 0', axis=1)
    return train, test 


def character_analysis(train, test):
    '''
    To analyze the performance of the technique that uses character level shingle.
    '''
    rocs_chr = {} # Dictionary of lists to store ROC of varying character sizes.
    # Combination of character and window sizes for singles.
    l1 = [2,4,6,8]          # Character sizes
    l2 = [13,15,17,19]      # Window sizes
    iterate = list(product(l1, l2))
    iterate = [(2,19)]
    for pair in iterate:
        start = time.time()
        rocs_win = {} # Dictionary to store ROC of varying window sized
        print(pair)
        # Characterize the "descriptions".
        train,test = charactizer(train,pair[0],pair[1]), charactizer(test,pair[0],pair[1])
        # Compute similarity
        train_df = similarity_classifier(train,128)
        test_df = similarity_classifier(test,128)
        # ROC analysis
        rocs_chr[(pair[0], pair[1])] = roc_analysis(train_df,test_df,rocs_win,pair[1])
        end = time.time()
        print("Completed simulating " + str(pair) + " in " + str(end-start) + " sec")

#    pickle.dump(rocs_chr,open('./data/rocs_v.pk',"wb"))
    # Confusion matrix analysis 
    cnf_mat(test_df)
    pdb.set_trace()

    return 

def token_analysis(train, test):
    '''
    To analyze the performance of the technique that uses word level shingle.
    '''
    cnf_mat(test_df)
    rocs = {} # Dictionary to store ROC
    for N in range(3,4):
        print(N)
        # Tokenize the "descriptions".
        train, test = tokenizer(train,N), tokenizer(test,N)
        # Compute similarity
        train_df = similarity_classifier(train,128)
        test_df = similarity_classifier(test,128)
        # ROC analysis
        roc_dict = roc_analysis(train_df,test_df,rocs,N)

    # Confusion matrix analysis 
    cnf_mat(test_df)
    pdb.set_trace()

    return 


def main():
    train, test = get_data()
    # Drop any rows with NaN (i.e. missing data)
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    # Get "match" data using 'id'; 0: no match 1: match
    train, test = get_match(train), get_match(test)
    # Analyse the technique of using shingles
    token_analysis(train, test)
    # Analyse the technique of using characters
    character_analysis(train, test)


    return 

if __name__ == '__main__':
    __spec__ = None # Required to run parrallel processing
    status = main()
    sys.exit()
