

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
        stock on the ith day, and an integer fee representing a transaction fee. 
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
        cash, hold = 0, -prices[0]
        for i in range(1,len(prices)):
            cash = max(cash, hold + prices[i] - fee)
            hold = max(hold, cash - prices[i])
            print("cash",cash)
            print("hold",hold)

        return cash

    def maxProfitBudget(self, present: list[int], future: list[int], budget: int) -> int:
        dp = [0] * (budget)
        for f,p in zip(future, present): 
            for bdg in range(len(dp),0,-1):
                dp[bdg-1] = max(dp[bdg-1], dp[bdg-p-1] + f - p) 

            a = 1
        b = 2
tmp = Solution()
#print(tmp.maxProfit([7,1,5,3,6,4]))
#print(tmp.maxProfit([1,2,3,4,5]))
#print(tmp.maxProfit([7,6,4,3,1]))
#print(tmp.maxProfitFee([1,4,5,3,4,2,7],2))
print(tmp.maxProfitBudget([1,2,3],[7,6,5],4))

