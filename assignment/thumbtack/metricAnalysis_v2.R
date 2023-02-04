
# -------- Invite-Quote trend analysis ----
# Author: Ranaji Krishna
# Date: July 13 2015.
# Notes: The code was developed to analyse trends across the entire data set.

# ---------- #


rm(list = ls(all = TRUE));  # clear all.
graphics.off();             # close all.

library("corpcor");         # Inverse of covarisanvce matrix.
library("gdata");           # Importing excel data.
library("xts");
library("data.table");
library("zoo");
library("mondate");
library("grDevices");
library("scales");
library("ggplot2");

require("xlsx");
require("lmtest");
require("bstats");          # White Test.
require("leave_out_cross_validation");
library("fOptions");        # For GBSOptions.
library("Matrix");
library("bbmle");
require("gamlss");

# ====== Data ===== #
setwd('/Users/vashishtha/myGitCode/thumbTack/');                    # Set directory
dataAll <- read.csv('universe_v2.csv', stringsAsFactor = TRUE);     # Import data.

# ----- Dataframe of Ratios ---- #
ratio_date <- data.frame(matrix(0,1,ncol=7));
colnames(ratio_date) <- c("in_st","sp_usr_id","in_qt_ratio","mod_ratio","hr","category_id","location_id");

# ----- Compute Ratios ----- #
for (i in 1 : max(dataAll$sp_usr_id) ){                                   # Iterate through service provider.
  
  tmp_sp <- subset(dataAll, dataAll$sp_usr_id == i);                      # Data of Service Provider.
  
  for (j in unique(floor(tmp_sp$time))){                                  # Iterate through time.
    
    tmp_hr <- subset(tmp_sp, (tmp_sp$time > (j) & tmp_sp$time < (j+1)));  # Data wihin the hour.
    if(dim(tmp_hr)[1]==0){
      next;                                                               # If no invites, then skip.
    }
    
    tmp_hr$in_qt_ratio <- (sum(tmp_hr$qt_st!='<null>')/dim(tmp_hr)[1]);   # Invite-Quote ratio.
    tmp_hr$mod_ratio <- (dim(tmp_hr)[1] - sum(tmp_hr$qt_st!='<null>'))/(dim(tmp_hr)[1] + sum(tmp_hr$qt_st!='<null>')); # **Not used!**
    tmp_hr$hr <- j + 1;                                                   # Track Hour.
    
    # ---- Populate dataframe ---- #
    ratio_date <- rbind(ratio_date, tmp_hr[1, c("in_st","sp_usr_id","in_qt_ratio","mod_ratio","hr","category_id","location_id")]); 
  }
  #browser()
}

browser();

ratio_date <- ratio_date[-1,];    # Remove top row.
ratio_date <- xts(ratio_date[,-1], order.by = as.POSIXct(ratio_date[,1])); # Order by Service provider user id.

# ---- Save variables --- #
# save(ratio_date,file="hr_ratio.Rda");
# save(dataAll,file="dataAll_qt_in_ratio.Rda");
# load(file="hr_ratio.Rda");
# load(file="dataAll_qt_in_ratio.Rda");
# ------

ep <- endpoints(ratio_date,'hours');              
tmp_ep <- period.apply(ratio_date[,2], ep, mean); # Mean of qoute-to-invite ratios within the hour.

# ---- Prepare data for Regression ----
tmp_data <- tmp_ep[,1];
tmp_data <- cbind(tmp_data,seq(1,1502));
colnames(tmp_data)[2] <- "hr";

# ---- GAMLSS Regression ----
reg_gam <- gamlss(in_qt_ratio ~ 0+hr, data=tmp_data,
                  family=BEINF(mu.link = "logit", sigma.link = "log", nu.link = "log", tau.link = "log"))

confint(reg_gam,"hr",level = .95);                            # Compte Cnfidence Interval. 

# ----- Generate Plots -----
pdf(file = ifelse(TRUE, "hr_dist.pdf"));           # Save plot.
hist(tmp_ep[,1], main="Distribution of Quote-to-Invite Ratio", 
     xlab="Ratio", ylab="Frequency", pch=19, cex=3);
dev.off();

pdf(file = ifelse(TRUE, "hr_plot.pdf"));           # Save plot.
plot(tmp_data[,1], main="Quote-to-Invite Ratio Vs Time", 
     xlab="Time", ylab="Ratio", pch=19, cex=3);
dev.off();

pdf(file = ifelse(TRUE, "scat_fit.pdf"));           # Save plot.
plot(as.numeric(tmp_data[,2]),as.numeric(tmp_data[,1]), xlab="Hours", ylab = "Quote-to-invite ratio", main="Fitted model")
lines(seq(1,1502),fitted(reg_gam),col="blue",pch=25,lwd=3)
dev.off();

my_res<- exp(reg_gam$residuals);
my_res<- my_res/(1+my_res);
pdf(file = ifelse(TRUE, "res_dist.pdf"));           # Save plot.
hist(my_res, main="Distribution of Residuals", 
     xlab="Residual", ylab="Frequency", pch=19);
dev.off();


# ----- Code for Example --- 
# sp=exp(reg_gam$mu.coefficients[1] * 18000);
# hist(fitted(reg_gam));
# my_res <- exp(reg_gam$residuals)/1+exp(reg_gam$residuals);
# hist(reg_gam$residuals)

# ----- Add Columns ---- #
# tmp_data <- data.frame(matrix(0, nrow = dim(dataAll)[1], ncol=4));
# colnames(tmp_data) <- c("in_qt_ratio","mod_ratio","hr","time");
# dataAll <- cbind(dataAll, tmp_data);


