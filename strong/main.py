
import sys
import pdb
import pandas as pd
import numpy as np

def get_data():
    buoy = pd.read_csv('data/buoy-data.csv')
    wide = pd.read_csv('data/wide.csv')
    pdb.set_trace()

    return 



def main():

    get_data()

    return 



if __name__ == '__main__':
    status = main()
    sys.exit()
