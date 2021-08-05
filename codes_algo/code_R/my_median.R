
partition <- function(tmpArray, st, ed, check_med){
  pvt <- tmpArray[ed];  # Pivot (convention: last value assigned).
  int1 <- st;           # Pivot Index. 
  while(st <ed){
    if(tmpArray[st] < pvt){
      m <- tmpArray[st];              # Implementing swap...
      tmpArray[st]  <- tmpArray[int1];
      tmpArray[int1] <- m;
      int1 <- int1 + 1;               # Index of next highest number.
    }
    st <- st + 1;    
  }
  
  tmpArray[ed] <- tmpArray[int1];     # Swap Pivot and highest number.
  tmpArray[int1] <-  pvt;
  return(list(tmpArray,int1));
}

my_median <- function(tmpArray, st, ed, check_pvt){
  if(st == 1){
    tmpArray_length <- length(tmpArray);
    if((tmpArray_length %%2) == 0){
      check_pvt <- 1 + (tmpArray_length /2);
    }else{
      check_pvt <- 0.5 + (tmpArray_length /2);
    }
  }
  if(st <= ed){ 
    x <- partition(tmpArray, st, ed);
    tmpArray <- x[[1]]; 
    int1 <- x[[2]];
    if(int1 > check_pvt){
      tmpArray <- my_median(tmpArray, st, int1-1, check_pvt);
    } else if (int1 < check_pvt){
      tmpArray <- my_median(tmpArray, int1+1, ed, check_pvt);
    } else{
      return(tmpArray[int1]);
    }
  }  
  return(tmpArray);
}

# set.seed(10);
# a <- as.array(round(runif(10,0,10)));
# b <- my_median(a,1,dim(a));
# print(a)
# print(b)
