
import sys
import pdb

class Solution:
    '''
    Given a m x n grid filled with non-negative numbers, find a path from 
    top left to bottom right, which minimizes the sum of all numbers along its path.
    Note: You can only move either down or right at any point in time.
    Example1:-
    Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
    Output: 7
    Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
    Example1:-
    Input: grid = [[1,2,3],[4,5,6]]
    Output: 12
    '''
    def minPathSum(self, grid: list[list[int]]) -> int:
        M, N = len(grid), len(grid[0]) 
        dp = [0] + [float('inf')] * (N-1)
        for i in range(M):
            dp[0] = dp[0] + grid[i][0]
            for j in range(1, N):
                dp[j] = min(dp[j-1], dp[j]) + grid[i][j]
        return dp[-1]

    def helper(self,A,i,j,dp):
        if i < 0 or j < 0: return float('-Inf')
        if i == 0 and j == 0: return A[i][j]
        if dp[i][j]!='-inf':  return dp[i][j]

        # Recursion.
        dp[i][j] = A[i][j] + max(self.helper(A,i-1,j,dp),self.helper(A,i,j-1,dp))
        return dp[i][j] 

    def minPathSumDP(self,A):
        '''
        Using dynamic programming with recursion and memoization.
        Compute 
                ∂(s,v) = max.(∂(s,u)+w(u,v)|(u,v) in E)
        The strategy computes the shotest path at all cells.
        '''
        n = len(A)
        # For memoization
        dp = [['-inf' for x in range(n)] for y in range(n)] # -Inf since max.
        return self.helper(A,n-1,n-1,dp)

tmp1 = Solution()
#A = [[1,3,1],[1,5,1],[4,2,1]]
A = [[1,-2,3,2,1],[1,2,4,-8,2],[2,1,6,4,3],[3,-7,1,0,-4],[4,3,2,2,1]]
#print(tmp1.minPathSum(A))
print(tmp1.minPathSumDP(A))

