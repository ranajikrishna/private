
rm(list = ls(all = TRUE));  # clear all.
graphics.off();             # close all.

library("corpcor");         # inverse of covarisanvce matrix.
library("gdata");           # importing excel data.
library("xts");
library("data.table");
library("zoo");
library("mondate");

require("xlsx");
require("lmtest");
require("bstats");          # White Test.
library("fOptions");        # For GBSOptions.


mySort <- function(tmpArray, int1, int2){
  
  if(int1==8)
  {browser();}  
  if(tmpArray[int1+1] < tmpArray[int1]){
    
    tmpVar <- tmpArray[int1+1]
    tmpArray[int1+1] <- tmpArray[int1];
    tmpArray[int1] <- tmpVar;
    int1 <- int1 + 1;
    mySort(tmpArray, int1, int2);
    
  }else if(int2 < length(col_a)){
    
    int2 <- int2 + 1;
    int1 <- int2;
    mySort(tmpArray, int1, int2);
    
  }
  
}

set.seed(10);
col_a <- round(runif(10,0,10))
mySort(col_a,1,1)

