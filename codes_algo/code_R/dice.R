

rm(list = ls(all = TRUE));  # clear all.
graphics.off();             # close all.


dice <- function(k,i){
  if (k < i){
    #browser()
    return (i*6)
  }
  else{
    return (dice(k, i + 1/6))
  } 
}

N = 100000
dieA <- runif(N)
dieB <- runif(N)
dif <- seq(1,N)
for (i in seq(1,N)){
  dif[i] = abs(dice(dieA[i],0) - dice(dieB[i],0))
}
print(mean(dif))
