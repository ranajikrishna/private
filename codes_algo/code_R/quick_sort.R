

partition <- function(tmpArray, st, ed){
  pvt <- tmpArray[ed];
  int1 <- st;
  while(st < ed){
    if(tmpArray[st] < pvt){
      m <- tmpArray[int1];
      tmpArray[int1] <- tmpArray[st];
      tmpArray[st] <- m;
      int1 <- int1 + 1;
    }
    st <- st + 1;
  }
  tmpArray[ed] <- tmpArray[int1];
  tmpArray[int1] <- pvt;
  return(list(tmpArray, int1));
}

quick_sort <- function(tmpArray, st, ed){
  if(st < ed){ 
    strArray <- partition(tmpArray, st, ed);
    tmpArray <- strArray[[1]]; int1 <- strArray[[2]];
    tmpArray <- quick_sort(tmpArray, st, int1 - 1);
    tmpArray <- quick_sort(tmpArray, int1 + 1, ed);
  }
  return(tmpArray)  
}

# set.seed(10);
# a <- as.array(round(runif(10,0,10)));
# b <- quick_sort(a,1,dim(a));
# print(a)
# print(b)
