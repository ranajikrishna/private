
import sys
import numpy as np
import pdb

class Solution:
    def maxProfit(self, prices: list[float]) -> float:
        '''
        You are given an integer array prices where prices[i] is the price 
        of a given stock on the ith day.On each day, you may decide to buy 
        and/or sell the stock. You can only hold at most one share of the 
        stock at any time. However, you can buy it then immediately sell it 
        on the same day. Find and return the maximum profit you can achieve.
        Example1:-
        Input: prices = [7,1,5,3,6,4]
        Output: 7
        Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), 
        profit = 5-1 = 4. Then buy on day 4 (price = 3) and sell on day 5 
        (price = 6), profit = 6-3 = 3. Total profit is 4 + 3 = 7.
        Example2:-
        Input: prices = [1,2,3,4,5]
        Output: 4
        Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), 
        profit = 5-1 = 4. Total profit is 4.
        Example3:-
        Input: prices = [7,6,4,3,1]
        Output: 0
        Explanation: There is no way to make a positive profit, so we never buy 
        the stock to achieve the maximum profit of 0.
        '''
        pre,profit = 0,0
        for cur,prc in enumerate(prices[1:]):
            if prc - prices[pre] > 0:
                profit = profit + prc - prices[pre] 
            pre += 1
        return profit

    def maxProfitFee(self, prices: list[float], fee: int) -> float:
        ''' 
        You are given an array prices where prices[i] is the price of a given 
        stock on the ith day, and an integer fee representing a trasaction fee. 
        Find the maximum profit you can achieve. You may complete as many 
        transactions as you like, but you need to pay the transaction fee for 
        each transaction. Note: You may not engage in multiple transactions 
        simultaneously (i.e., you must sell the stock before you buy again).
        Example1:-
        Input: prices = [1,3,2,8,4,9], fee = 2
        Output: 8
        Explanation: The maximum profit can be achieved by:
            - Buying at prices[0] = 1
            - Selling at prices[3] = 8
            - Buying at prices[4] = 4
            - Selling at prices[5] = 9
            The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
        '''

        '''
        Explanation:-
        The initial holding price is hold=p[0]. At i+1 if stock price p[i+1] > p[i] we 
        should hold, and update the cash value as p[i+1]-p[0]-f. Else, we should
        examine if the hold price needs to be updated from hold=-p[0]. Hold is 
        *not* updated if, p[i]-p[0]-f + p[i+2]-p[i+1]-f < p[i+2]-p[0]-f 
        => p[i]-p[i+1]-f < 0 => p[i]-p[0]-f-p[i+1] < -p[0]. Otherwise, hold is 
        update to p[i]-p[0]-f-p[i+1] ≡ cash(previous txn.) - p[i+1](new hold).
        Therefore, to compute the new cash, we do hold + p[i+2] - f.  
        '''
        cash, hold = 0, -prices[0]
        for i in range(1,len(prices)):
            cash = max(cash, hold + prices[i] - fee)
            hold = max(hold, cash - prices[i])
            print("cash",cash)
            print("hold",hold)

        return cash

    def maxProfitBudget(self, present: list[int], future: list[int], budget: int) -> int:
        '''
        You are given two 0-indexed integer arrays of the same length present 
        and future where present[i] is the current price of the ith stock and 
        future[i] is the price of the ith stock a year in the future. You may 
        buy each stock at most once. You are also given an integer budget 
        representing the amount of money you currently have.
        Return the maximum amount of profit you can make.
        Example1:-
        Input: present = [5,4,6,2,3], future = [8,5,4,3,5], budget = 10
        Output: 6
        Explanation: One possible way to maximize your profit is to:
        Buy the 0th, 3rd, and 4th stocks for a total of 5 + 2 + 3 = 10.
        Next year, sell all three stocks for a total of 8 + 3 + 5 = 16.
        The profit you made is 16 - 10 = 6.
        It can be shown that the maximum profit you can make is 6.
        Example2:-
        Input: present = [2,2,5], future = [3,4,10], budget = 6
        Output: 5
        Explanation: The only possible way to maximize your profit is to:
        Buy the 2nd stock, and make a profit of 10 - 5 = 5.
        It can be shown that the maximum profit you can make is 5.
        Example3:-
        Input: present = [3,3,12], future = [0,3,15], budget = 10
        Output: 0
        Explanation: One possible way to maximize your profit is to:
        Buy the 1st stock, and make a profit of 3 - 3 = 0.
        It can be shown that the maximum profit you can make is 0.
        '''
        def sub_prob(index,budget):
            if ((len(present)-1==index)|(budget<0)): 
                if (budget<0): 
                    dp[(index,budget)] = -1*profit[index]
                    return 
                dp[(index,budget)] = 0 
                return

            sub_prob(index+1,budget) 
            sub_prob(index+1,budget-present[index+1])
        
            dp[(index,budget)] = max(
                    dp[(index+1,budget)],
                    dp[(index+1,budget-present[index+1])] + profit[index+1]
                    ) 
            return 

        dp = {}  
        profit = np.array(future)-np.array(present)
        sub_prob(0,budget)
        sub_prob(0,budget-present[0])
        return max(dp[(0,budget)], dp[(0,budget-present[0])]+profit[0])

#        dp = [0] * (budget)
#        for f,p in zip(future, present): 
#            for bdg in range(len(dp),0,-1):
#                dp[bdg-1] = max(dp[bdg-1], dp[bdg-p-1] + f - p) 
#
#            a = 1
#        b = 2

def knapSack(W, wt, val, n):
    '''
    :param W: capacity of knapsack 
    :param wt: list containing weights
    :param val: list containing corresponding values
    :param n: size of lists
    :return: Integer
    '''
    # code here
    if n == 0 or W == 0:
        return 0
    if wt[n-1] <= W:
        return (max(val[n-1]+knapSack(W-wt[n-1], wt, val, n-1), knapSack(W, wt, val, n-1)))
    else:
        return (knapSack(W, wt, val, n-1))

tmp = Solution()
#print(tmp.maxProfit([7,1,5,3,6,4]))
#print(tmp.maxProfit([1,2,3,4,5]))
#print(tmp.maxProfit([7,6,4,3,1]))
#print(tmp.maxProfitFee([1,4,6,5,14,2,7],2))
#print(tmp.maxProfitBudget([2,2,5],[3,4,10],6))
#print(tmp.maxProfitBudget([5,4,6,2,3],[8,5,4,3,5],10))
print(knapSack(10,[5,4,6,2,3],[3,1,-2,1,2],5))

