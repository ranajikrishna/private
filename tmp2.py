

import sys
import pdb


def get_data():

    data = {}

    with open('abnormal_ai.csv', mode ='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)

        for line in csvFile:
            pdb.set_trace()


    return 



def main():

    get_data()

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
