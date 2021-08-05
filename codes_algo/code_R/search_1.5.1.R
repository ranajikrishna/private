
find_k1 <- function(tmpArray, k, l, r, j, int1, int2){
  if(is.na(tmpArray[[1]][r]) == "FALSE"){
    if(k > tmpArray[[1]][r]){
      if(j==0){
        int1 <- int1 + 1;
        r <- 2^int1;
        find_k1(tmpArray, k, l, r, j, int1, int2);
        return(find_k1(tmpArray, k, l, r, j, int1, int2));
      }
      if (j == 1){
        m <- l;  
        l <- r;
        r <- 2*r-m;
        find_k1(tmpArray, k, l, r, j, int1, int2);
      }
    }  else if (k == tmpArray[[1]][r]){
      return(list(tmpArray[[1]][r],tmpArray[[2]][r]));
    } else {
      j <- 1;
      r <- floor((l + r)/2);
      if(r==l && k != tmpArray[[1]][r]){
        return(-1)
      }
      find_k1(tmpArray, k, l, r, j, int1, int2);
    }
  } else {
    int2 = int2 + 1;
    r <- 2^int1 - int2;
    find_k1(tmpArray, k, l, r, j, int1, int2);
    return(list(tmpArray[[1]][r],tmpArray[[2]][r]));
  }
}

search_1.5.1 <- function(tmpArray, k){
  source('~/AlgoPrep/myCodes/bubble_sort.R');
  col_a <- bubble_sort(tmpArray);
  r <- 1;
  l <- 0;
  j <- 0;
  int1 <- 0;
  int2 <- 0;
  return(find_k1(col_a, k, l, r, j, int1, int2));
}

