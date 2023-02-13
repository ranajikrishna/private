import sys
import pdb

class Solution:
    '''
    A subarray of a 0-indexed integer array is a contiguous non-empty sequence 
    of elements within an array. The alternating subarray sum of a subarray that 
    ranges from index i to j (inclusive, 0 <= i <= j < nums.length) is
    nums[i] - nums[i+1] + nums[i+2] - ... +/- nums[j].
    Given a 0-indexed integer array nums, return the maximum alternating subarray 
    sum of any subarray of nums.

    Example1:-
    Input: nums = [3,-1,1,2]
    Output: 5
    Explanation:
    The subarray [3,-1,1] has the largest alternating subarray sum.
    The alternating subarray sum is 3 - (-1) + 1 = 5.

    Example2:-
    Input: nums = [2,2,2,2,2]
    Output: 2
    Explanation:
    The subarrays [2], [2,2,2], and [2,2,2,2,2] have the largest alternating 
    subarra sum.
    The alternating subarray sum of [2] is 2.
    The alternating subarray sum of [2,2,2] is 2 - 2 + 2 = 2.
    The alternating subarray sum of [2,2,2,2,2] is 2 - 2 + 2 - 2 + 2 = 2.

    Example3:-
    Input: nums = [1]
    Output: 1
    Explanation:
    There is only one non-empty subarray, which is [1].
    The alternating subarray sum is 1.

    NOTE: Odd indices are always +ve and evens are always -ve.
    '''
    def maximumAlternatingSubarraySum(self, nums):
        pos_sum, neg_sum, max_val = float("-inf"), float("-inf"), float("-inf")
        for num in nums:
            pos_sum, neg_sum = max(neg_sum+num,num), pos_sum-num
            max_val = max(max_val,pos_sum,neg_sum)
        return max_val

tmp = Solution()
A = [1,2,3,4,5] 
A = [-5,1,3,4,-2] 
print(tmp.maximumAlternatingSubarraySum(A))

