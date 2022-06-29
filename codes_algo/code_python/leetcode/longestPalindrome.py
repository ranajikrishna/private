
import pdb

class Solution:
    '''
    Manacher algorithm
    Explanation: https://www.youtube.com/watch?v=nbTSfrEfo6M
    '''

    def longestPalindrome(self, s):
        # Transform S into T.
        # For example, S = "abba", T = "^#a#b#b#a#$".
        # ^ and $ signs are sentinels appended to each end to avoid bounds checking
        T = '#'.join('@{}$'.format(s))
        n = len(T)
        P = [0] * n
        C = R = 0
        for i in range (1, n-1):
            mirror = 2*C - 1
            P[i] = (R > i) and min(R - i, P[mirror]) # equals to i' = C - (i-C)
            # Attempt to expand palindrome centered at i
            while T[i + 1 + P[i]] == T[i - 1 - P[i]]:
                P[i] += 1

            # If palindrome centered at i expand past R,
            # adjust center based on expanded palindrome.
            if i + P[i] > R:
                C, R = i, i + P[i]

        # Find the maximum element in P.
        maxLen, centerIndex = max((n, i) for i, n in enumerate(P))
        return s[(centerIndex  - maxLen)//2: (centerIndex  + maxLen)//2]


def main():
    t = "cbaxabaxabybaxabyb"
    chk = Solution()
    ans = chk.longestPalindrome(t) 
    pdb.set_trace()
    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
