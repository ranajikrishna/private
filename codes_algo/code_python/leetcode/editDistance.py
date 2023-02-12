
import sys
import pdb

class Solution:
    '''
    Given two strings word1 and word2, return the minimum number of operations 
    required to convert word1 to word2.

    You have the following three operations permitted on a word:

    Insert a character
    Delete a character
    Replace a character

    Example1:-
    Input: word1 = "horse", word2 = "ros"
    Output: 3
    Explanation: 
        horse -> rorse (replace 'h' with 'r')
        rorse -> rose (remove 'r')
        rose -> ros (remove 'e')
    Example2:-
    Input: word1 = "intention", word2 = "execution"
    Output: 5
    Explanation: 
        intention -> inention (remove 't')
        inention -> enention (replace 'i' with 'e')
        enention -> exention (replace 'n' with 'x')
        exention -> exection (replace 'n' with 'c')
        exection -> execution (insert 'u')
    '''
    def minDistance(self, word1: str, word2: str) -> int:
        memo = {}
        
        def dfs(i, j):
            if i == 0 or j == 0: return j or i
                        
            if (i,j) in memo:
                return memo[(i,j)]
            
            if word1[i] == word2[j]:
                ans = dfs(i-1, j-1)
            else: 
                ans = 1 + min(dfs(i, j-1), dfs(i-1, j), dfs(i-1, j-1))
                #         Insert(to right)   Delete       Replace
                
            memo[(i,j)] = ans
            return memo[(i,j)]
        
        return dfs(len(word1), len(word2))

def main():
    word1 = 'abc'
    word2 = 'abe'
    tmp = Solution()
    print(tmp.minDistance(word1, word2))
    pdb.set_trace()

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
