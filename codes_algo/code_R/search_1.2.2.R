

search_1.2.2 <- function(tmpArray, k){
  source('~/AlgoPrep/myCodes/bubble_sort.R');
  col_a <- bubble_sort(tmpArray);
  u <- length(tmpArray);
  l <- 0;
#   browser()
  while(l <= u){
    m <- floor((l + u)/2);
    if(col_a[[1]][m]<k){
      l <- m + 1;
    }else if(col_a[[1]][m]==k){
      return (list(col_a[[1]][m],col_a[[2]][m]));
    }else{
      u <- m -1;
    }
  }
  return(-1);
}

