
import sys
import pdb

def maxSum(A):
    def findPath(dp,n):
        '''
        Use maximum path matrix (dp) to determine the path to traverse from along
        the negative slope direction, from source (botton right) to destination
        (top left).
        '''
        path = []
        r,c = n-1,n-1
        for k in range(2*n-2):
            if r==0: a = 0
            elif c==0: a = 1
            # Follow path to maximum path value.
            # Note: In max(a,b), index 0 is returned if a==b since horizontal 
            # movement (keep r constant and change c) is given priority over 
            # vertical movement.
            else: a = [dp[r][c-1],dp[r-1][c]].index(max(dp[r][c-1],dp[r-1][c]))
            
            if a==0: c -= 1 # Keep row constant, change col.  
            else: r -= 1    # Keep col constant, change row.
            path.append(a)
        return path 

    def helper(A,i,j,dp):
        '''
        Using dynamic programming with recursion and memoization.
        Compute 
                ∂(s,v) = min.(∂(s,u)+w(u,v)|(u,v) in E)
        The strategy computes the shotest path at all cells.
        '''
        if i<0 or j<0: return float('-inf') 
        if i==0 and j==0: return A[0][0]  
        if dp[i][j] != float('-inf'): return dp[i][j] # Since max., use '-inf'.
       
        h,v = helper(A,i-1,j,dp) , helper(A,i,j-1,dp) 
        dp[i][j] = A[i][j] + max(h,v)
        return dp[i][j] 

    n= len(A)
    dp = [[float('-inf') for j in range(n)] for i in range(n)] # For memoization.
    max_sum = helper(A,n-1,n-1,dp) # Compute max. path sum. 
    return [max_sum, findPath(dp,n)]

def main():
#    A = [[1,-2,3,2,1],[1,2,4,-8,2],[2,1,6,4,3],[3,-7,1,0,-4],[4,3,2,2,1]]
    A = [[1,3,1],[1,5,1],[4,2,1]]
    top_left,top_right,btm_right,btm_left = [],[],[],[]
    n = len(A)//2 + 1
    # Generate 4 sub-matrices from original matrix.
    for row in A[:n]:
        top_left.append(row[:n])
        top_right.append(row[n-1:])
    for row in A[n-1:]:
        btm_right.append(row[n-1:])
        btm_left.append(row[:n])

    # Transform sub-matirces so that source-destination are in neg. slope direction. 
    # where source is in bottom right and destination is in top left corners.
    top_left = top_left[::-1]
    top_left = [top_left[r][::-1] for r in range(n)]
    top_right = top_right[::-1]
    btm_left = [btm_left[r][::-1] for r in range(n)]
    path,max_sum = {},[]
    for quadrant,sub_matrix in enumerate([top_left,top_right,btm_right,btm_left]):  # Iterate in required priority.
        # Get max. sum path value and direction in terms of hor. and vert. movements.
        path[quadrant] = maxSum(sub_matrix) 
        # Generate directions. 
        # Note: We traverse from bottom-right to top-left
        # corner of the sub-matrix in terms of horrizontal(0) and vertical (1) 
        # steps. We need to translate this into left-right and up-down movements
        # accordingly.
        if quadrant==0: path[quadrant].append([*map(lambda x: 'd' if x==1 else 'r',path[quadrant][1])])
        if quadrant==1: path[quadrant].append([*map(lambda x: 'd' if x==1 else 'l',path[quadrant][1])])
        if quadrant==2: path[quadrant].append([*map(lambda x: 'u' if x==1 else 'l',path[quadrant][1])])
        if quadrant==3: path[quadrant].append([*map(lambda x: 'u' if x==1 else 'r',path[quadrant][1])])
        max_sum.append(path[quadrant][0]) # Max sum path.

    # Print path to max. sum.
    for quadrant,item in path.items():
        if item[0]==max(max_sum):
            print(item[2])
            break;
    pdb.set_trace()

if __name__ == '__main__':
    status = main()
    sys.exit()
