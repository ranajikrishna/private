
import sys
import pdb

from sys import maxsize 
 
 
def maxSubArraySum(a, size):
 
    max_so_far = -maxsize - 1
    max_ending_here = 0
 
    for i in range(0, size):
        max_ending_here = max_ending_here + a[i]
        if (max_so_far < max_ending_here):
            max_so_far = max_ending_here
 
        if max_ending_here < 0:
            max_ending_here = 0
    return max_so_far

def max_sub_array(A):
    '''
    Sub-array with the maximum sum. 
    Example:1
    A = [-10,5,1,-1,7,3,-1,-2]
    Output: 15
    Example:2
    A = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6 
    Example:3
    A = [-3,-5,-3,-5,-7,-4,-6]
    NOTE:-
    A = [7, 4, 1, -15, 7, 3, -1, -2]
                   *
    All subarray sums that include (i-1)th elmt. will be increased by the ith 
    elmt. to yield the subarray sums that include the ith elmt. Hence the max. 
    sum from the subarrays that include the (i-1)th elmt. is retained as the max. 
    So we only need to keep a record of this max.
    At * point the global max = 12, and the current max is reset to 0. This is
    coz. the max. sum of any subarray to the left that includes * is -ve, and so
    including it in subarrays to the right of * will only reduce the value of the
    max. sum subarray. Hence current max is reset to 0.
    
    https://medium.com/@rsinghal757/kadanes-algorithm-dynamic-programming-how-
    and-why-does-it-work-3fd8849ed73d
    '''

    glo_max,cur_max = float('-inf'),0
    for i in A:
        cur_max += i
        if glo_max < cur_max: glo_max = cur_max
        if cur_max < 0: cur_max = 0
    return (glo_max)


def main():
    A = [-10,5,8,-8,7,3,1,2]
    A = [-2,1,-3,4,-1,2,1,-5,4]
    A = [-2, -3, 4, -1, -2, 1, 5, -3]
    A = [-3,-5,-3,5,-7,-4,-6,-5]
    
    print(max_sub_array(A))
    print(maxSubArraySum(A,len(A)))

    return

if __name__ == '__main__':
    status = main()
    sys.exit()
