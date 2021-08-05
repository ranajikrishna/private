
# find_k <- function(tmpArray, int1){
#   if(tmpArray[[1]][int1] == tmpArray[[2]][int1]){
#     return(tmpArray[[1]][int1]);
#   }else if(int1 <= length(tmpArray[[1]])){
#     int1 = int1 + 1;
#     find_k(tmpArray, int1)
#   } else{
#     return(-1);
#   }  
# }

# search_1.4 <- function(tmpArray){
#   source('~/AlgoPrep/myCodes/bubble_sort.R');
#   browser()
#   array_ind <- abs(tmpArray - seq(1,length(tmpArray)))
#   col_a <- bubble_sort(array_ind);
#   return(find_k(col_a, 1));
# }

search_1.4 <- function(tmpArray){
  source('~/AlgoPrep/myCodes/bubble_sort.R');
  array_ind <- abs(tmpArray - seq(1,length(tmpArray)))
  col_a <- bubble_sort(array_ind);
  if(col_a[[1]][1]==0){
    return(col_a[[2]][1]);
  } else{
    return(-1);
  }
}



