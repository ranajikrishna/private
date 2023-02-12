# -------- Flight data analysis ----
# Author: Ranaji Krishna
# Date: July 13 2015.
# Notes: The code was developed tosort flight data.

# ---------- #


rm(list = ls(all = TRUE));  # clear all.
graphics.off();             # close all.

library("gdata")           # Importing excel data.
library("xts")
library("data.table")
library("zoo")
library("mondate")
library("grDevices")
library("scales")
library("ggplot2")
require("xlsx")
library("Matrix")
library("stringi")
library("stringr")

myMed <- function (tmpListA, tmpListB){
  #browser()
  if(dim(tmpListA)==2){
    #browser()
    return (median(c(tmpListA,tmpListB)))
  }
  
  posA <- ceiling(dim(tmpListA)/2)
  posB <- ceiling(dim(tmpListB)/2)
  
  medA <- tmpListA[posA]
  medB <- tmpListB[posB]
  
  retListA <- tmpListA[posA : dim(tmpListA)]
  retListB <- tmpListB[1 : posB]
  
  if (medA < medB){
    myMed(retListA, retListB)
  }else{
    myMed(retListB, retListA)
  }
  
} 

N <- 4
a <- as.array(sort(sample(1:100,N)))
b <- as.array(sort(sample(1:100,N)))

myMedian <- myMed(a, b)

print (myMedian)
