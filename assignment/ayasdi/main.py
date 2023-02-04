
import sys
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import pdb


def double_word(var):

    tmp = []
    for i in range(0,len(var)):
        tmp += var[i] + var[i]

    return tmp 


def viz_pca(data):

    pdb.set_trace()
    return 



def corr_mat(data,colname):
  
    ax = sns.heatmap(
    np.corrcoef(data.T), 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)

    pdb.set_trace()
    ax.set_xticklabels(
    ax.get_xticklabels(colname),
    rotation=45,
    horizontalalignment='right');
    
    plt.show()

    return 



def load_data(filename):

    data = []
    label = []
    colname = []
    bad_data=[]
    with open(filename, newline='') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        i = 0
        for row in tsvin:
            if i == 0:
               colname = row
               i += 1
               continue;
            try:
                data.append(list(map(float,row[:-1])))
                label.append(row[-1])
            except:
                bad_data.append(row)

    return np.array(data),np.array(label),colname,bad_data


def main(argv=None):

    filename = '/Users/vashishtha/myGitCode/code_ayasdi/dataset_challenge_one.tsv'
    data,label,colname,bad_data = load_data(filename)
    double_word(['h','e','l','l','o'])
#    viz_pca(data)
#    corr_mat(data[:,0:35],colname)
    pdb.set_trace()

    return


if __name__ == "__main__":
    status = main()
    sys.exit(status)
