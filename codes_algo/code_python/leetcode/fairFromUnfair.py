
import random
import pdb

def fair(pair):
'''
Given an unfair coin, where probability of HEADS coming up is P and TAILS 
is (1-P), implement a fair coin from the given unfair coin.
'''
    if pair==('H','T') or pair==('T','H'):
        return pair[0]

def flip(p):
    return 'H' if random.random() < p else 'T'

def main():
    N = 100000
    p = 0.6
    flips = list()
    [flips.append((flip(p),flip(p))) for _ in range(1,N)]
    fair_flips = list(map(fair, flips))
    tmp = [i for i in fair_flips if i!=None]

    pc = sum([i=='H' for i in tmp])/len(tmp)
    pdb.set_trace()
    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
