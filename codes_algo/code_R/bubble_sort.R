# 
large <- function(duplet){
  #browser() 
  if(duplet[[1]][1] > duplet[[1]][2]){
    return(list(rev(duplet[[1]]),rev(duplet[[2]])));
  }else{
    return(list(duplet[[1]],duplet[[2]]));
  }
  
}

exchange <- function(tmpArray){
  for(i in seq(1, length(tmpArray[[1]])-1)){
     tmpList <- large(list(tmpArray[[1]][i:(i+1)],tmpArray[[2]][i:(i+1)]));
     #browser()
     
     tmpArray[[1]][i:(i+1)] <- tmpList[[1]];
     tmpArray[[2]][i:(i+1)] <- tmpList[[2]];
  }
  return(tmpArray);
}

start_sort <- function(tmpArray){
  newArray <- exchange(tmpArray);
  if(isTRUE(all.equal(tmpArray[[1]],newArray[[1]]))){
    return(newArray);
  }else{
    browser()
    return(start_sort(newArray));
  }
}

bubble_sort <- function(tmpSeq){
  tmpArray <- list(tmpSeq,seq(1,length(tmpSeq)));
  return(start_sort(tmpArray));
}

# set.seed(10);
 a <- round(runif(10,0,10))
 b <- bubble_sort(a)
 