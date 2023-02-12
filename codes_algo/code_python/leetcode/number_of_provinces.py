
import sys
import pdb

def findProvinces(A,s):
    '''There are n cities. Some of them are connected, while some are not. 
    If city a is connected directly with city b, and city b is connected
    directly with city c, then city a is connected indirectly with city c.

    A province is a group of directly or indirectly connected cities and no 
    other cities outside of the group.

    You are given an n x n matrix isConnected where isConnected[i][j] = 1 if
    the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

    Return the total number of provinces.
    '''

    def bfs(A,s):
        queue = [s]
        while queue:
            city = queue.pop(0)
            for i,j in enumerate(A[city]):
                if j==1 and i not in visited:
                    visited.append(i)
                    queue.append(i)
        return

    visited,province = [],0
    for city in range(len(A)):
        if city not in visited:
            bfs(A,city)
            province += 1
    return province


def main():
    A=[[0,0,0,1,1,0,0,0,0],[0,0,0,1,0,0,1,0,0],[0,0,0,0,0,1,0,0,1],[1,1,0,0,0,0,1,0,0],
       [1,0,0,0,0,0,0,0,0],[0,0,1,0,0,1,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1],
       [0,0,1,0,0,0,0,1,0]]

    n = len(A)

    findProvinces(A,0)

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
