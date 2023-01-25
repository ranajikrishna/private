
import sys
from collections import defaultdict
import pdb


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

#    def addEdge(self,n,v):
#        self.graph[n].append(v)

    def addEdge(self,vert_edge:list[int]):
        for (u,v) in vert_edge: 
            self.graph[u].append(v)

def BFS(graph:Graph,s:int) -> dict:
    
    visited = [s] 
    queue = [s]
    parent = defaultdict(list)
 
    while queue:
 
        # Dequeue a vertex from
        # queue and print it
        s = queue.pop(0)
        print (s, end ="\n")
 
        # Get all adjacent vertices of the
        # dequeued vertex s. If a adjacent
        # has not been visited, then mark it
        # visited and enqueue it
        for i in graph[s]:
            if  i not in visited:
                queue.append(i)
                visited.append(i)
                parent[s].append(i)

    return parent

def main():
    g = Graph()
    graph_structure = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]
    g.addEdge(graph_structure)
#    g.addEdge(0, 1)
#    g.addEdge(0, 2)
#    g.addEdge(1, 3)
#    g.addEdge(1, 4)
#    g.addEdge(2, 5)
#    g.addEdge(2, 6)

#    print(g)

    print(BFS(g.graph,0))

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
