

my_union <- function(tmpArray1, tmpArray2){
  
  tmpArray1 <- sort(tmpArray1)
  tmpArray2 <- sort(tmpArray2)  
  
  size1<- length(tmpArray1)
  size2<- length(tmpArray2)
  itr1 <- 1
  itr2 <- 1
    
    while(itr1 <= size1){
      
      if(tmpArray2[itr2] > tmpArray1[itr1]){
        itr1 = itr1 + 1
      }else if(tmpArray2[itr2] < tmpArray1[itr1]){
        tmpArray1 <- append(tmpArray1, tmpArray2[itr2], itr1-1)
        itr1 <- itr1 + 1
        itr2 <- itr2 + 1
        size1 <- size1 + 1
        break
      }else {
        itr1 <- itr1 + 1
        itr2 <- itr2 + 1
        break
      }

    }
    if (itr2 != size2){
    tmpArray1 <- as.array(c(tmpArray1,tmpArray2[itr2:size2]))
    itr2 <- size2 + 1
    itr1 <- size1 + 1  
    }
  return(tmpArray1)
}


# tmp1 <- as.array(c(1, 2, 3, 4, 6,7,90))
# tmp2 <- as.array(c(300,5,800))
# 
# print(my_union(tmp1,tmp2))

# si1 <- 6
# si2 <- 3
