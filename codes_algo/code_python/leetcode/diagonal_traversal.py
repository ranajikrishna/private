

import sys
import pdb
import collections


def findDiag(mat: list[list[int]]) -> list[int]:

    diag = collections.defaultdict(list)
    M = len(mat)
    N = len(mat[0])

    for c in range(M):
        for r in range(N):
            diag[c+r].append(mat[c][r])

    res = []
    a=-1
    for k,v in diag.items():
        res.append(v[::a])
        a = -1 * a
    
    pdb.set_trace()
    return 

class Solution:
    def findDiagonalOrder(self, mat: list[list[int]]) -> list[int]:
        pdb.set_trace()
        dic = collections.defaultdict(list)
        m,n = len(mat), len(mat[0])
        for i in range(m):
            for j in range(n):
                dic[i+j].append(mat[i][j])
                
        res = []
        
        direction = -1
        
        for num in range(n*m):
            res.extend(dic[num][::direction])
            direction*=-1
    
        return res


def main():
    mat = [[1,2,3],[4,5,6],[7,8,9]]

    findDiag(mat)
    tmp = Solution()
    tmp.findDiagonalOrder(mat)



    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
