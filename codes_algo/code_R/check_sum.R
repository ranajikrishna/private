
check_sum <- function(tmpArray, K){
  
  tmpArray <- sort(tmpArray);
  r <- length(tmpArray);
  l <- 1;
  i <- 0;
  storePair <- array(list(NULL),c(10,1));
  
  while(l < r){
    if(tmpArray[l] + tmpArray[r] < K){
      l <- l+1;
    }else if(tmpArray[l] + tmpArray[r] > K){
      r <- r-1;
    }else{
      i <- i + 1;
      storePair[[i]] <- list(l,r);
      r <- r - 1;
    } 
  }
  return(storePair);
}


tmp1 <- as.array(c(1, 2, 3, 4, 6,7,90))
k = 10

print(check_sum(tmp1,k))