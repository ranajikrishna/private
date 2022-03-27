import sys
import pdb

class Solution5:
    def reverse(self, x: int) -> int:
        num2str = str(abs(x))
        str_list = [i for i in num2str]
        rev_num =  "".join(str_list[::-1])
        num = int(rev_num) if x>=0 else -1*int(rev_num)
        return num if -2**31 < num < 2**31-1 else 0     

class Solution: 
    def reverse(self, x):                               
        y = str(x)[::-1]                                    
        x = int('-' + y.strip('-')) if x < 0 else int(y)      
        return x if -2**31 < x < 2**31-1 else 0    

def main():
    num = 1534236469
    tmp = Solution5()
    print(tmp.reverse(num))
    return 
        
        
if __name__ == '__main__':
    status = main()
    sys.exit()
