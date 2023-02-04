
import sys
import pdb
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addWeightEdge(self, array):
        for (u,v,w) in array:
            self.graph[u].append((v,w))

def bellmanford():

    return 


def main():
    #A = [(0,1,8),(0,2,3),(1,2,1),(1,3,2),(2,1,4),(2,3,8),(2,4,2),(3,4,7),(4,3,9)]
    A = [('A','B',8),('A','C',3),('B','C',1),('B','D',2),('C','B',4), \
         ('C','D',8),('C','E',2),('D','E',7),('E','D',9)]
    g = Graph()
    g.addWeightEdge(A)
    print(bellmanford(g.graph,'A'))

if __name__ == '__main__':
    status = main()
    sys.exit()
