
import pdb
import sys


def trap(height: list([int])) -> int:
    '''
    LeetCode: https://leetcode.com/problems/trapping-rain-water/
    YouTube: https://www.youtube.com/watch?v=C8UjlJZsHBw
    '''

    vol = 0
    left_max = height[0]
    right_max = height[-1]
    left_ind = 1
    right_ind = len(height) - 2
    while (right_ind - left_ind) >= 0:
        if (left_max < right_max):
            i = left_ind
            if height[i] > left_max:
                left_max = height[i]
                continue;
            max_ = left_max
            left_ind += 1
        else:
            i = right_ind
            if height[i] > right_max:
                right_max = height[i]
                continue;
            max_ = right_max
            right_ind -= 1
        vol += max_ - height[i] 

    pdb.set_trace()
    return vol 

def main():
    height = [4,2,0,3,2,5]
    height = [0,1,0,2,1,0,1,3,2,1,2,1]
    print(trap(height))

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
