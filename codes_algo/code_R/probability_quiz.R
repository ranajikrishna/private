#Simulate 10 throws with a starting amount of x=money=10
#n=10
simulate = function(){
  #money won/lost in a single game
  money = 1
  for(i in 1:2){
    if(runif(1) < 0.5)
      money = money/2
    else
      money = 2*money
  }
  return(money)
}

#The Money vector keeps track of all the games
#N is the number of games we play
N = 2
Money = numeric(N)
for(i in 1:N)
  Money[i]= simulate()

mean(Money);median(Money)
#Probabilities
#Simulated
table(Money)/1000
#Exact
2^{-10}*choose(10,10/2)

#Plot the simulations
plot(Money)