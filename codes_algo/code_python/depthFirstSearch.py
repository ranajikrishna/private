
import sys
from collections import defaultdict
import pdb

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self,vert_edge:list[int]):
        for pair in vert_edge:
            self.graph[pair[0]].append(pair[1])

def DFSUtil(graph:Graph,visited:list[int],s:int):
        visited.append(s)
        print(s,end='\n')
        for i in graph[s]:
            if i not in visited:
                DFSUtil(graph,visited,i)

def DFS(graph:Graph,s:int):
        visited = []
        DFSUtil(graph,visited,s)

def main():
    g = Graph()
    graph_structure = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]
    g.addEdge(graph_structure)

    DFS(g.graph,0)

    return 



if __name__ == '__main__':
    status = main()
    sys.exit()
