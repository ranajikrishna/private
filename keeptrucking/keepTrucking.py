import sys 
import pdb
import numpy as np


def maximumSafeTraversal(logbook):
    '''
    KeepTruckin
    Given ["BG", "CA", "FI", "OK"] compute the length of the longest seq. A-I
    '''

    pdb.set_trace()
    N = len(logbook)
    M = 26
    str_stp_idx = [[ord(i[0]) & 31, ord(i[1]) & 31] for i in logbook]
    bin_mat = np.array([[0] * M] * N)


    for i,j in enumerate(str_stp_idx):
        if j[0]>j[1]:
            j[0],j[1] = j[1],j[0]
        bin_mat[i][j[0]:j[1]] = 1

    seq = sum(bin_mat)
    k = 0
    size = list([])
    for i in seq:
        if i != 0 :
            k += 1
        elif k != 0:
            size.append(k)
            k = 0
    return max(size)-1


def main():

    logbook = ["BG", "CA", "FI", "OK"]
    print(maximumSafeTraversal(logbook))
    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
