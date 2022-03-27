import sys
import pdb

class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        x_str = str(x)
        n = len(x_str)
        if int(x_str) == int(x_str[::-1]):
            return True
        else: 
            return False
def main():
    num = 12345654321

    tmp = Solution()
    print(tmp.isPalindrome(num))
    return 
        
        
if __name__ == '__main__':
    status = main()
    sys.exit()
