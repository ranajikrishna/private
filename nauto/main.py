

import sys
import pdb


def order_list_freq(input: list[str]) -> list[str]:

    word = dict.fromkeys(input,0)
    for key in input:
        word[key] += 1

    return sorted(word)[::-1]

#def order_list_freq(input: list[str]) -> list:
#    '''
#    Given a list of strings, order them in decreasing frequency.
#    '''
#    
#    freq = dict.fromkeys(input, 0)
#    for term in input:
#        freq[term] += 1
#    
#    freq_list = []
#    for key,value in freq.items():
#       freq_list.append((value,key))
#
#    freq_list.sort()
#    freq_sort = []
#    for i in freq_list[::-1]:
#        freq_sort.append(i[1])
#
#    return freq_sort

def main():
    input = ['dog','cat','horse','dog','cat','dog','horse', 'cat', 'cat']
    print(order_list_freq(input))
    return 

if __name__ == '__main__':
    status = main()
    sys.exit()


