
rm(list = ls(all = TRUE));  # clear all.
graphics.off();             # close all.

hash_1.9 <- function(tmpArray, K){
  h <- hash();
  for(int1 in seq(1, length(tmpArray))){
      browser()
    h[[as.character(tmpArray[int1])]] <- int1;
    cmt <- K - tmpArray[int1];
    if (has.key(as.character(cmt), h)){
      #browser()
      return(list(h[[as.character(cmt)]],int1));
    }
  }
}
