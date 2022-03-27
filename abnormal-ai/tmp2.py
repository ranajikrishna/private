

import sys
import csv
import pdb


def get_data():

    data = {}

    with open('abnormal_ai.csv', mode ='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)

        for line in csvFile:

            if line[0] in data.keys():
                mean = data[line[0]][0]
                freq = data[line[0]][1]
                if line[2] == "\'open\'":
                    update = -1*(int(line[1]))
                else:
                    update = int(line[1])
                data[line[0]][0] = (mean * freq + update)/(freq+0.5)
                data[line[0]][1] += 0.5
            else:
                if line[2] == "\'open\'":
                    data[line[0]] = [-1*int(line[1])*2,0.5]
                else:
                    data[line[0]] = [int(line[1])*2,0.5]


    for i,j in enumerate(data):
        pdb.set_trace()

    return 



def main():

    get_data()

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
