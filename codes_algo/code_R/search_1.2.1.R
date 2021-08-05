
find_k <- function(tmpArray, k, l, r){
  #browser()
  if(k > tmpArray[[1]][l]){
    l <- floor((l+r)/2);
    if(l==r && k != tmpArray[[1]][l]){
      return(-1);
    }
    find_k(tmpArray, k, l, r);
  } else if (k == tmpArray[[1]][l]){
    return(list(tmpArray[[1]][l],tmpArray[[2]][l]));
  } else if (k < tmpArray[[1]][l]){
    m = r;  
    r = l;
    l = 2*r-m+1;
    if(l==r && k != tmpArray[[1]][l]){
      return(-1);
    }
    find_k(tmpArray, k, l, r);
  } else {
    return(-1);
  }  
}

search_1.2.1 <- function(tmpArray, k){
  source('~/AlgoPrep/myCodes/bubble_sort.R');
  col_a <- bubble_sort(tmpArray);
  r <- length(tmpArray);
  l <- 1;
  return(find_k(col_a, k, l, r));
}

