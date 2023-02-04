
import sys 
import pdb
import numpy as np


def mst(logbook: list[str]) -> int:
    tup = []
    for pair in logbook:
        fst = ord(pair[1])&31 if ord(pair[0])&31 > ord(pair[1])&31 else ord(pair[0])&31  
        scd = ord(pair[0])&31 if ord(pair[0])&31 > ord(pair[1])&31 else ord(pair[1])&31 
        tup.append((fst,scd))

    tup = sorted(tup)
    max_dist = tup[0][1] - tup[0][0]
    max_pair = tup[0]
    for pair in tup[1:]:
        if ((pair[0] < max_pair[1]) & (pair[1] > max_pair[1])):
            max_dist = pair[1] - max_pair[0]
            max_pair = (max_pair[0],pair[1])
        elif pair[0] > max_pair[1]:
            max_dist = max_dist if max_dist>pair[1]-pair[0] else pair[1]-pair[0]
            max_pair = max_pair if max_dist>pair[1]-pair[0] else pair

    return max_dist 




def mst1(logbook: list[str]) -> int:

    tup = []
    for pair in logbook:
        fst = ord(pair[1])&31 if ord(pair[0])&31 > ord(pair[1])&31 else ord(pair[0])&31  
        scd = ord(pair[0])&31 if ord(pair[0])&31 > ord(pair[1])&31 else ord(pair[1])&31 
        tup.append((fst,scd))

    tup_srt = sorted(tup)
    trav = []
    i = 0
    start = tup_srt[0][0]
    for pair in tup_srt[1:]:
        if pair[0] < tup_srt[i][1]:
            i += 1
            continue
        else:
            end = tup_srt[i][1]
            trav.append(end-start)
            start = pair[0]
            i += 1

    return max(trav) 




def maxSafeTravel(logbook: list[str]) -> int:
    
    bin_book = []
    for ind,pair in enumerate(logbook):
        bin_mat = [0]*26
        if ord(pair[0])&31 < ord(pair[1])&31:  
            bin_mat[(ord(pair[0])&31)-1:(ord(pair[1])&31)+1]=np.ones((ord(pair[1])&31) - (ord(pair[0])&31)+1)
        else:
            bin_mat[(ord(pair[1])&31)-1:(ord(pair[0])&31)+1]=np.ones((ord(pair[0])&31) - (ord(pair[1])&31)+1)
        bin_book.append(bin_mat)
    sum_bk = np.sum(bin_book,0)
    res,i=[],0
    for chk in sum_bk:
        if chk!=0:
            i+=1
        else:
            res.append(i)
            i=0
    return max(res) 


def maximumSafeTraversal(logbook):
    '''
    KeepTruckin
    Given ["BG", "CA", "FI", "OK"] compute the length of the longest seq. A-I
    '''

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
    print(mst(logbook))
#    print(maxSafeTravel(logbook))
#    print(maximumSafeTraversal(logbook))
    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
