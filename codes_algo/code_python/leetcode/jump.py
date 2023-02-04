
class Solution:
    def canJump(self, nums: List[int]) -> bool:
    '''
    You are given an integer array nums. You are initially positioned 
    at the array's first index, and each element in the array represents your 
    maximum jump length at that position.
    Return true if you can reach the last index, or false otherwise.
    Example1:-
    Input: nums = [2,3,1,1,4]
    Output: true
    Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
    Example2:-
    Input: nums = [3,2,1,0,4]
    Output: false
    Explanation: You will always arrive at index 3 no matter what. Its maximum 
    jump length is 0, which makes it impossible to reach the last index.
    '''
        farthest = 0    
        for idx, val in enumerate(nums):
            if idx > farthest:
                return False
            if idx + val > farthest:
                farthest = idx + val 
        return True

def jump1(q):
    N = len(q)
    itr = 1
    good = True
    for i in range(N-2,0,-1):
        if q[i] < itr:
            itr += 1
            good = False
        else:
            itr = 1
            good = True

    return good 

#print(jump([9,4,2,1,0,2,0]))
#print(jump([2,4,2,1,0,2,0]))
print(jump([0,1]))
