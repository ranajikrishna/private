
import sys

class Solution:
    def minPathSum(self, grid: List(List[int])) -> int:
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
        M, N = len(grid), len(grid[0]) 
        dp = [0] + [float('inf')] * (N-1)
        for i in range(M):
            dp[0] = dp[0] + grid[i][0]
            for j in range(1, N):
                dp[j] = min(dp[j-1], dp[j]) + grid[i][j]
        return dp[-1]


q = [[1,3,1],[1,5,1],[4,2,1]]
print(minPathSum(q))

