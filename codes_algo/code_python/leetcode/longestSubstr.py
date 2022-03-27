
import sys
import pdb

class Solution:
    '''
    Given a string s, find the length of the longest substring without repeating 
    characters.
    '''

    def lengthOfLongestSubstring(self, s: str) -> int:    
        start = 0
        end = 0
        max_len = 0
        d={}
        while end<len(s):
            if s[end]  in d and d[s[end]] >= start:
                start = d[s[end]]+1
            max_len = max(max_len , end-start+1)
            d[s[end]] = end
            end+=1
        return(max_len)    

def main():

    tmp = Solution()
    string = 'pwwkewikl'
    print(tmp.lengthOfLongestSubstring(string))
    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
