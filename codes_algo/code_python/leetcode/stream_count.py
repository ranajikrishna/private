
import sys
import pdb

def sampleStats(count: list([int])) -> list([float]):
    '''
    LeetCode: https://leetcode.com/problems/statistics-from-a-large-sample/
    '''
    s = valid_cnt = mode = 0
    min_val = max_val = None
    for i,val in enumerate(count):
		# valid --> if count of element k is greater than 0
        if val>0:
			# total elements
            valid_cnt += val
			# find sum of all elements
            s += i*val
			# update min value only for the first occurence
            if min_val is None:
                min_val = i
			# update max value till last occurence
            max_val = i
			# if current element count is greater than previous element, update mode(initial mode is set to 0)
            if val>count[mode]:
                mode = i
	# calculate mean
    mean = s/valid_cnt
	# find median --> if total elements are even--> median is avg of middle 2, if odd--> middle element 
    k = valid_cnt//2
    if valid_cnt%2!=0:
        median = median_finder(k+1, count)
    else:
        median = (median_finder(k, count) + median_finder(k+1, count))/2
    return list(map(float, [min_val, max_val, mean, median, mode]))

# helper function to find kth element
def median_finder(k, count):
    valid_cnt = 0
    for i,val in enumerate(count):
        if val>0:
            valid_cnt += val
            if valid_cnt>=k:
                    return i
def main():
    count = [0,1,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    print(sampleStats(count))
    return

if __name__ == '__main__':
    status = main()
    sys.exit()
