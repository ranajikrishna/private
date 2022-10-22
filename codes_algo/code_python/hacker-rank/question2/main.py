
import pdb
import sys
import numpy as np
import pandas as pd


def encryptionValidity(instructionCount, validityPeriod, keys):

    srt_key = sorted(keys)
    div = []
    for idx,item in enumerate(srt_key):
        itr,sum_ = 0,0
        while itr <= idx:
            sum_ += int(item%srt_key[itr]==0)
            itr+=1
        
        div.append(sum_)

    max_div = max(div) * (10**5)
    return [int(((instructionCount*validityPeriod)/(max_div))>1), max_div]



def main():
    ins_cnt = 9677958
    val_prd = 50058356
    keys = [83315,22089,19068,64911,67636,4640,80192,98971]
    tmp = encryptionValidity(ins_cnt, val_prd, keys)

    pdb.set_trace()

    return

if __name__ == '__main__':
    status = main()
    sys.exit()
