

rm(list = ls(all = TRUE));  # clear all.
graphics.off();             # close all.

amount <- matrix(100,(2^7),1);
amount[2^7] <- 1/100;
comb_array <- combos(7)$binary;
for(i in 1: (2^7-1)){
  
  for(j in seq(1,7)){
    if(comb_array[i,j]==1){
      amount[i] <- amount[i] + 1;
    }else{
      amount[i] <- 1/amount[i];
      
    }
  }
}

fv <- sum(amount)/2^7;
