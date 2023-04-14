
import sys
import pdb
import collections as cp

class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1 



def coin_change1(arr:list[int], money:int) -> int:
    
    def dynamic_count(arr, money, dp):
        if money < 0:  return -1
        if money == 0: return 0
        for num in arr:
            if money < num: count =  -1
            elif dp[money-num] != float('inf'): count = dp[money-num]
            else: count = dynamic_count(arr, money-num, dp)
            if count != -1: 
                count += 1
                dp[money] = min(dp[money],count)
        return dp[money] 
    
    count = dynamic_count(arr,money,[float('inf')]*(money+1))
    return -1 if count==float('inf') else count

def main():
    money = 3
    arr = [1,2]
    tmp = Solution()
    tmp.coinChange(arr,money)
    pdb.set_trace()
    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
