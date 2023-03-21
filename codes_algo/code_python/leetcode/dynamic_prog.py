
import sys
import pdb

def min_cost_climbing_stairs(stairs: list[int]) -> int:
    stairs.append(0)
    dp = [stairs[0],stairs[1]]
    for i in range(2,len(stairs)):
        dp.append(min(dp[i-1],dp[i-2]) + stairs[i])
    return dp.pop()


def min_cost_climbing_stairs_theory(stairs: list[int]) -> int:
    dp = [0] * (len(stairs) + 1)
    step = len(stairs)
    def compute_cost(step, dp):
        if step < -1: return 0
        dp[step] = min(compute_cost(step-1,dp),compute_cost(step-2,dp)) + stairs[step] 
    return 

def climbing_stairs_theory(height,dp):
    if height >= 0:
        if height == 0 : return 1
        if dp[height] == -1:
            dp[height] = min(climbing_stairs(height-1,dp) + climbing_stairs(height-2,dp) 
            return dp[height]
        else: return dp[height] 
    else: return 0

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

    min_cost_climbing_stairs([1,100,1,1,1,100,1,1,100,1])


    pdb.set_trace()



if __name__ == '__main__':
    status = main()
    sys.exit()
