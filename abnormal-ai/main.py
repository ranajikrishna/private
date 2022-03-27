

import sys
import csv
import pdb

def get_data():
    '''
    Given a stream of data (abnormal-ai.csv), generate the mean of the values
    grouped by the id.
    '''
    data_dict = {}
    # opening the CSV file
    with open('abnormal_ai.csv', mode ='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        # displaying the contents of the CSV file
        for line in csvFile:
            pdb.set_trace()
            if line[0] in data_dict.keys():
                mean = data_dict[line[0]][0]
                freq = data_dict[line[0]][1]
                if line[2]=="\'open\'":
                    data_dict[line[0]][0] = (mean * freq - int(line[1]))/(freq+0.5) 
                else:
                    data_dict[line[0]][0] = (mean * freq + int(line[1]))/(freq+0.5)
                data_dict[line[0]][1] += 0.5
            else:
                if line[2]=="\'open\'":
                    data_dict[line[0]] = [-1*int(line[1])*2,0.5]
                else:
                    data_dict[line[0]] = [int(line[1])*2,0.5]

    return data_dict


def main():
    get_data()

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
