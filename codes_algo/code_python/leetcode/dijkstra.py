
import sys
from collections import defaultdict
import heapq
import pdb

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addWeightEdge(self,array:list):
        for (u,v,w) in array:
            self.graph[u].append((v,w))

def dijkstra(graph:Graph, S:str) -> list[int]:
    '''
    Generates the shortest path from vertex S to *all* the other vertices that 
    can be reached from vertex S.
    '''
    priority_queue = [(0,S)]
    shortest_path = {}
    while priority_queue:
        # Get the vertex with the minimum weight, i.e. greedy.
        w,v = heapq.heappop(priority_queue) 
        if v not in shortest_path:      # Go to the vertex if not visited.
            shortest_path[v] = w            
            # Relax the vertex, i.e. compute all the edge weights.
            for v_i, w_i in graph[v]:       
                # Compute total weight and append it to the priority queue.
                heapq.heappush(priority_queue, (w + w_i, v_i)) 

    return shortest_path

def main():
    #A = [(0,1,8),(0,2,3),(1,2,1),(1,3,2),(2,1,4),(2,3,8),(2,4,2),(3,4,7),(4,3,9)]
    A = [('A','B',8),('A','C',3),('B','C',1),('B','D',2),('C','B',4), \
         ('C','D',8),('C','E',2),('D','E',7),('E','D',9)]
    g = Graph()
    g.addWeightEdge(A)
    print(dijkstra(g.graph,'A'))

if __name__ == '__main__':
    status = main()
    sys.exit()
