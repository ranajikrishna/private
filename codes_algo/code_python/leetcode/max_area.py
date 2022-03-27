
import sys
import pdb

def maxArea(height: list([int])) -> int:
    '''
    LeetCode: https://leetcode.com/problems/container-with-most-water/
    YouTube: https://www.youtube.com/watch?v=oLBeYtjMa-s
    '''
    left_ind = 0
    right_ind = len(height)-1
    area = []
    while (right_ind-left_ind)>=1:   
        left_val = height[left_ind]
        right_val = height[right_ind]
        area.append(min(left_val,right_val)*(right_ind-left_ind))
        if left_val < right_val:
            left_ind +=1
        else:
            right_ind -= 1
    return max(area) 

def main():
    height = [1,8,6,2,5,4,8,3,7]
    height = [1,1]
    height = [4,3,2,1,4]
    height = [1,2,1]
    print(maxArea(height))

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()

