##VIX Adjusted Momentum Strategy v.1

#clear environment and console
rm(list = ls()) 
cat("\014")

#set working directory: path that the input files are located, and the output files are saved
#setwd("/Users/vashishtha/myGitCode/quant/george")
setwd("/Users/ranajikrishna/invoice2go/git_code/ranaji_q/quant/george/")

#install and load required packages
#install.packages('tseries')
#install.packages('reshape2')
#install.packages('PerformanceAnalytics')
#install.packages('fBasics')
#install.packages('psych')
library(tseries)
library(reshape2)
library(PerformanceAnalytics)
library(fBasics)
library(psych)

#read in data file (first column is assumed to contain dates, second column to contain VIX levels, 
#the other columns contain share prices)
#sp500 is the name of the input file
sp500 = read.csv('S&P500_VIX.csv') 
sp500 = sp500[,1:12]


# Separate dates from stock prices
colnames(sp500)[1] = 'date'
sp500$date = as.Date(sp500$date,format='%m/%d/%Y')
date = sp500$date
sp500$date = NULL

# Convert zero prices to NAs
sp500[sp500 == 0] = NA

# Input assumptions for holding period, rolling window 
window = 252        # 1 trading year
holdTime = 20       # Train window
form_time <- 90     # Trade window 
skip_time <- 10     # Skip time 
executionCost = 200 # average s&p bid-ask spread + 1.0% (bps)
completeObs = 125   # must have 125 complete observations to proceed

# Track time
ptm <- proc.time() 
ord = matrix(ncol = ncol(sp500))
gm = matrix(ncol = ncol(sp500))    

ln = log(sp500[,2:ncol(sp500)]) 
ln[is.na(ln)] <- 0

ret = as.matrix(diff(as.matrix(ln, lag=1, difference=1)))
VIX = as.vector(sp500[1:(nrow(sp500)-1),1])
adj_ret = ret / VIX 

#initiation of variables used in the loop
gm = matrix(nrow = ((nrow(sp500) - window) + 1), ncol = ncol(adj_ret))

for (j in 1 : ncol(adj_ret))
{
  for (i in 1 : ((nrow(adj_ret) - form_time) + 1) )
    {
    gm[i, j] = mean(adj_ret[(i : (window + i - 1)), j])
    #gm[i, j] = adj_ret[form_time + i - 1]/adj_ret[i] - 1
    
  }
}

#order = apply(gm, 1, sort)
ord = sort(gm, na.last = NA, decreasing = TRUE)
winners = head(ord, 3)
losers = tail(ord, 3)

browser()
        
          
        
        
        