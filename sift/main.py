#Given a list of coordinates, an origin and a count k, return the top k elements that are closest to the origin.
#coordinates = [ [1, 1], [1, 2], [2, 2]],
  #  origin = [0, 0],
    #k = 2
    # [[3,3],[5,-1],[-2,4]], k = 2

import sys
import pdb
import numpy as np

def distance(cord,org,k):
    
    dist = {}
    #dist = dict.fromkeys(str(pair),0)

    for pair in cord:
        dist[str(pair)] = np.sqrt((pair[0] - org[0])**2 + (pair[1] - org[1])**2)
          
    pdb.set_trace()
    return sorted(dist,keys=(dist.values()),reverse=True)

def main():
    
    #coordinates = [[1, 1], [1, 2], [2, 2]]
    coordinates = [[3,3],[5,-1],[-2,4]]
    origin = [0,0]
    k = 2
    print(distance(coordinates,origin,k))
    return 


if __name__ == '__main__':
    status = main()
    sys.exit()


