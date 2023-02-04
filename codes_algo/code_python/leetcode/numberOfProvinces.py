
import collections
from collections import deque

class Solution:
    def findCircleNum(self, isConnected: list[list[int]]) -> int:
    def dfs(start):
    '''There are n cities. Some of them are connected, while some are not. 
    If city a is connected directly with city b, and city b is connected
    directly with city c, then city a is connected indirectly with city c.

    A province is a group of directly or indirectly connected cities and no 
    other cities outside of the group.

    You are given an n x n matrix isConnected where isConnected[i][j] = 1 if
    the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

    Return the total number of provinces.
    '''
        for end in range(len(isConnected)):
            if end not in visited:
                visited.append(end)
                dfs(end)

    visited = []
    numOfProvinces = 0
    for start in range(len(isConnected)):
        if start not in visited:
            numOfProvinces += 1
            dfs(start)

    return numOfProvinces

q = [[1,1,0],[1,1,0],[0,0,1]]
q = [[1,0,0],[0,1,0],[0,0,1]]
q = [[1,1,0],[0,1,0],[0,0,1]]
q = [[1,0,0,1],[0,1,1,0],[0,1,1,1],[1,0,1,1]]

print(findCircleNum(q))        
