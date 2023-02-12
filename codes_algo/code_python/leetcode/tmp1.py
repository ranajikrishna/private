

class Solution:
    def minDistance(word1,word2):

        if i==0 or j==0: return i or j

        if (i,j) in memo: return memo[(i,j)]

        if word1[i-1]==word2[j-1]:
            ans = dfs(i-1,j-1)
        else:
            ans = 1 + min(dfs(i,j-1),dfs(i-1,j),dfs(i-1,j-1))

        memo[(i,j)] = ans

        return ans 



tmp = Solution()

tmp.minDistance('horse','ros')
