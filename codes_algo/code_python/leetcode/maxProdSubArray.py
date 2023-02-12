
class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        '''Given an integer array nums, find a subarray that has the largest
        product, and return the product. The test cases are generated so that 
        the answer will fit in a 32-bit integer.
        Example1:-
        Input: nums = [2,3,-2,4]
        Output: 6
        Explanation: [2,3] has the largest product 6.
        Example2:-
        Input: nums = [-2,0,-1]
        Output: 0
        Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
        '''
        if len(nums)==1: return nums[0]
        min_val,max_val=[nums[0]],[nums[0]]
        i = 1
        for cur in nums[1:]:
            min_val.append(min(cur*max_val[i-1],cur*min_val[i-1],cur))
            max_val.append(max(cur*min_val[i-1],cur*max_val[i-1],cur))
            i += 1
        return max(max_val)


tmp = Solution()
#t = [2,3,-2,4]
t = [-2,0,-1]
t = [-4,-3]
print(tmp.maxProduct(t))

