
import sys
import pdb

def generateMatrix(n):
    A, lo = [[n*n]], n*n
    while lo > 1:
        print(A)
        lo, hi = lo - len(A), lo
        A = [list(range(lo, hi))] + list(zip(*A[::-1]))
    print(A)
    return A

def main():
    tmp = generateMatrix(3)
    pdb.set_trace()

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()

