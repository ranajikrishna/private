
import sys
import pdb


class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right


def house_robber(tree: TreeNode) -> int:
    ''' 
    You are a professional robber planning to rob houses along a street. Each 
    house has a certain amount of money stashed, the only constraint stopping 
    you from robbing each of them is that adjacent houses have security systems 
    connected and it will automatically contact the police if two adjacent houses
    were broken into on the same night.

    Given an integer array nums representing the amount of money of each house,
    return the maximum amount of money you can rob tonight without alerting the
    police.

    Example1:-
    Input: nums = [1,2,3,1]
    Output: 4
    Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
    Total amount you can rob = 1 + 3 = 4.

    Example2:-
    Input: nums = [2,7,9,3,1]
    Output: 12
    Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob 
    house 5 (money = 1).
    Total amount you can rob = 2 + 9 + 1 = 12.
    '''
    def compute_money(node):
        if (node == None or node.val == None): return [0,0]
        incl = node.val + compute_money(node.left)[1] + compute_money(node.right)[1]
        excl = max(compute_money(node.left)[0] + max(compute_money(node.right)[0],compute_money(node.right)[1]),
                   compute_money(node.right)[0] + max(compute_money(node.left)[0],compute_money(node.left)[1]))
        return [incl,excl]
    return max(compute_money(tree))
    


def min_cost_climbing_stairs_theory(stairs: list[int]) -> int:
    dp = [-1] * (len(stairs) + 1)
    step = len(stairs)
    stairs.append(0)
    def compute_cost(step):
        if step <= -1: return 0
        if dp[step] != -1: return dp[step]
        dp[step] = min(compute_cost(step-1),compute_cost(step-2)) + stairs[step] 
        return dp[step] 
    return compute_cost(step)

def min_cost_climbing_stairs(stairs: list[int]) -> int:
    '''
    You are given an integer array cost where cost[i] is the cost of ith step 
    on a staircase. Once you pay the cost, you can either climb one or two steps.
    You can either start from the step with index 0, or the step with index 1.
    Return the minimum cost to reach the top of the floor.

    Example1:-
    Input: cost = [10,15,20]
    Output: 15
    Explanation: You will start at index 1.
        - Pay 15 and climb two steps to reach the top. 
    The total cost is 15.

    Example2:-
    Input: cost = [1,100,1,1,1,100,1,1,100,1]
    Output: 6
    Explanation: You will start at index 0.
        - Pay 1 and climb two steps to reach index 2.
        - Pay 1 and climb two steps to reach index 4.
        - Pay 1 and climb two steps to reach index 6.
        - Pay 1 and climb one step to reach index 7.
        - Pay 1 and climb two steps to reach index 9.
        - Pay 1 and climb one step to reach the top.
    The total cost is 6.
    '''
    stairs.append(0)
    dp = [stairs[0],stairs[1]]
    for i in range(2,len(stairs)):
        dp.append(min(dp[i-1],dp[i-2]) + stairs[i])
    return dp.pop()

def climbing_stairs_theory(height,dp):
    if height < 0:  return 0
    if height == 0: return 1
    dp[height] = climbing_stairs_theory(height-1,dp) + \
            climbing_stairs_theory(height-2,dp)
    return dp[height] 

def climbing_stairs(height: int) -> int:
    ''' 
    You are climbing a staircase. It takes n steps to reach the top. Each time 
    you can either climb 1 or 2 steps. In how many distinct ways can you climb 
    to the top?
    Example 1:-
    Input: n = 2
    Output: 2
    Explanation: There are two ways to climb to the top.
    1. 1 step + 1 step
    2. 2 steps

    Example 2:-
    Input: n = 3
    Output: 3
    Explanation: There are three ways to climb to the top.
    1. 1 step + 1 step + 1 step
    2. 1 step + 2 steps
    3. 2 steps + 1 step
    '''
    if height==1: return 1
    dp =[1]*(height+1)
    for i in range(2,height+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[height]

def main():
    # --- Climbing stairs ---
#    tmp = climbing_stairs_theory(5,[-1]*6)
#    tmp = climbing_stairs(5)

    # --- Climbing stairs with cost ---
#    print(min_cost_climbing_stairs_theory([1,100,1,1,1,100,1,1,100,1]))
#    print(min_cost_climbing_stairs_theory([2,1,3,4]))
    
    # --- House robber ---
#    root = [3,4,15,1,3,20,1]
#    def populate_tree(i, n):
#        if i < n:
#            node = TreeNode()
#            node.val = root[i]
#            node.left = populate_tree(2*i+1, n) 
#            node.right = populate_tree(2*i+2, n) 
#            return node
#        return None
#
#    bin_tree = populate_tree(0, len(root))
#    print(house_robber(bin_tree))



    return


if __name__ == '__main__':
    status = main()
    sys.exit()
