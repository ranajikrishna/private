"""

https://www.hackerrank.com/challenges/correlation-and-regression-lines-8/problem

"""
def var_cov(x,y):
    mean_x = mean(x)
    mean_y = mean(y)
    return sum([(x[i]-mean_x)*(y[i] - mean_y) for i in range(len(x))])
    
def mean(data):
    return sum(data)/len(data)

def slope(ind, dep):
    cov = var_cov(ind,dep)
    var = var_cov(ind,ind)
    return round(cov/var,3)

phy = [15, 12, 8, 8, 7, 7, 7, 6, 5, 3]
his = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]
print(slope(phy,his))
