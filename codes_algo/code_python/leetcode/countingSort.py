import sys
import pdb

def countingSort2(arr: list[int]) -> list[int]:
    
    freq = [0] * (max(arr) + 1)             # Frequency of elements.
    for value in arr:
        freq[value] += 1

    rank_index,cml_sum = [freq[0]],freq[0]  # Starting index of ranked elements
    for value in freq[1:]:
        cml_sum += value                    # Cummulative sum.
        rank_index.append(cml_sum)
    
    output = [0] * len(arr) 
    for value in arr[::-1]:
#       output[rank_index[value]-1] = value        # Ascending order 
        output[len(arr)-rank_index[value]] = value # Descending order
        rank_index[value] -= 1

    return output
                     
def countingSort(arr: list[int]) -> list[int]:
    L = [0] * (max(arr) + 1)
                     
    for value in arr :
        L[value] += 1

    output = []
    for i in range(len(L)):
        output.extend([i] * L[i])

    return output


def main():

    # A = [2,1,4,5,1,1,2,3,4,5]
     A = [3,5,7,5,5,3,6]
     print(countingSort2(A))
     return 

if __name__ == "__main__":
    status = main()
    sys.exit()
