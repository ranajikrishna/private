
import sys
import pdb


def decode(s: str, dp: dict={}) -> int:

    if (s in dp.keys()): return dp[s]
    if (int(s[-1])==0 and int(s[-2:])>26): return 0
    if int(s[0]) == 0: return 0
    if (int(s) < 11 or int(s)==20): return 1
    if (int(s) < 27): return 2
    if (len(s)<3): return 1

    if (int(s[0:2]) < 27): 
        comb = decode(s[1:],dp) + decode(s[2:],dp)
    else: 
        comb = decode(s[1:],dp) 
    dp[s] = comb
    return comb 


def main():
    s = '230'
    dp = {}
    print(decode(s,dp))

    return

if __name__ == '__main__':
    status = main()
    sys.exit
