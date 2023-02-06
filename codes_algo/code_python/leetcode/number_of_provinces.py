
import sys
import pdb

def findProvinces(A,s):
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
    pdb.set_trace()
    return


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
