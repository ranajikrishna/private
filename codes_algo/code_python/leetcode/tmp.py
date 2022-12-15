
import sys
import pdb
from operator import itemgetter

def probOfHead(prob: list[float], target: int) -> float:
    dp = [1] + [0]*target
    for index, p in enumerate(prob):
        for k in range(min(index+1, target), -1, -1):
            dp[k] = (dp[k-1] if k else 0) * p + dp[k] * (1 - p)
    return 


def main():

    markdict = {"Tom":67, "Tina": 54, "Akbar": 87, "Kane": 43, "Divya":73}
    #marklist = sorted(markdict.items(), key=lambda x:x[1])
    marklist = sorted(markdict.items(), key=itemgetter(1))
    sortdict = dict(marklist)
    print(sortdict)


#    target = 5
#    prob = [0.1,0.3,0.8,0.4,0.7,0.9,0.6, 0.5]
#    print(probOfHead(prob, target))

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()



