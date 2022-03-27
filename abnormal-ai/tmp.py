

import sys
import csv
import pdb
import collections

def get_data():
    stat = collections.defaultdict(list)
    pdb.set_trace()
    with open('abnormal_ai.csv', mode='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            if line[2] == '\'open\'':
                tmp = 0 
    return 


def main():

    get_data()

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
