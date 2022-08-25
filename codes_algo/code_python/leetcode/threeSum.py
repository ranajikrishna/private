import pdb
import sys

import numpy as np


def twoSum2(nums: list[int], target: int) -> list[int]:
    res = []
    nums.sort()
    l,r = 0,len(nums)-1 
    while(l<r):
        sums = nums[l] + nums[r]
        if sums<target:
            l+=1
        elif sums>target:
            r-=1
        else:
            res.append([nums[l],nums[r]])
            l+=1
            while (l<r and nums[l]==nums[l-1]):
                l+=1
            l+=1
    return res 


def threeSum(nums: list[int],target: int) -> list[int]:

    res = []
    nums.sort()
    for i,k in enumerate(nums):
        if i>0 and k==nums[i-1]:
            continue;
        l,r = i+1,len(nums)-1
        while(l < r):
            sums = k + nums[l] + nums[r]
            if sums>0:
                r-=1
            elif sums<0:
                l+=1
            else:
                pdb.set_trace()
                res.append([k,nums[l],nums[r]])
                l+=1
                while (l<r and nums[l]==nums[l-1]):
                    l+=1

    return res

def main():
    
    nums = [-1,0,1,2,-1,-4]
    nums = [3,0,1,2,3,1,-4]
    nums = [0,0,0]
    print(threeSum(nums,0))

    return 


if __name__ == "__main__":
    status = main()
    sys.exit()
