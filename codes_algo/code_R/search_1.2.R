
find_k <- function(tmpArray, k, l, r){
  
  if(k < tmpArray[[1]][r]){
    r <- floor((l + r)/2);
    if(r==l && k != tmpArray[[1]][r]){
      return(-1)
    }
    find_k(tmpArray, k, l, r);
  } else if (k == tmpArray[[1]][r]){
    return(list(tmpArray[[1]][r],tmpArray[[2]][r]));
  } else if (k > tmpArray[[1]][r]){
    m = l;  
    l = r;
    r = 2*r-m;
    if(r==l && k != tmpArray[[1]][r]){
      return(-1)
    }
    find_k(tmpArray, k, l, r);
  } else {
    return(-1);
  }  
}

search_1.2 <- function(tmpArray, k){
  source('~/AlgoPrep/myCodes/bubble_sort.R');
  col_a <- bubble_sort(tmpArray);
  r <- length(tmpArray);
  l <- 0;
  return(find_k(col_a, k, l, r));
}
