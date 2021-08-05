# -------- Invite-Quote trend analysis ----
# Author: Ranaji Krishna
# Date: Sept. 22 2015.
# Notes: The code was developed to analyse trends within Category and Location.
#   
# ---------- #


rm(list = ls(all = TRUE));  # clear all.
graphics.off();             # close all.

library("corpcor");         # inverse of covarisanvce matrix.
library("gdata");           # importing excel data.
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
setwd('/Users/vashishtha/myGitCode/thumbTack/');                        # Set directory
dataAll <- read.csv('universe_v2.csv', stringsAsFactor = TRUE);         # Import data.


# ----- Dataframe to store coefficient for each Category or Location ----
size_count <- length(unique(dataAll$location_id));                      # Number of Categories (or Locations).
str_coef <- data.frame(matrix(0, size_count, ncol=5));                  
colnames(str_coef) <- c("Location_id","Coefficient","Std. Error", "p-value","Location");

for (k in seq(1, size_count)){                                          # Iterate through Categories (or Location).
  #print (k);  
  # ----- Dataframe of Ratios ---- #
  ratio_date <- data.frame(matrix(0,1,ncol=7));
  colnames(ratio_date) <- c("in_st","sp_usr_id","in_qt_ratio","mod_ratio","hr","category_id","location_id");
  
  tmp_cat <- subset(dataAll, dataAll$location_id==k);         # Data for Category (or Location). 
  uniq_sp <- unique(tmp_cat$sp_usr_id);                       # Id's of unique service providers in the Category (or Location).
  #  uniq_sp <- uniq_sp[order(uniq_sp),];
  for (i in uniq_sp ){                                        # Iterate through service provider.
    
    tmp_sp <- subset(tmp_cat, tmp_cat$sp_usr_id == i);        # Data of service provider in the Category (or Location).
    
    for (j in unique(floor(tmp_sp$time))){                    # Iterate throgh time.
    
      tmp_hr <- subset(tmp_sp, (tmp_sp$time > (j) & tmp_sp$time < (j+1)));  # Data of service provider in the hour.
      if(dim(tmp_hr)[1]==0){
        next;                                                 # If no invites, then skip.
      }
      
      tmp_hr$in_qt_ratio <- (sum(tmp_hr$qt_st!='<null>')/dim(tmp_hr)[1]);   # Compute Ratio.
      tmp_hr$mod_ratio <- (dim(tmp_hr)[1] - sum(tmp_hr$qt_st!='<null>'))/(dim(tmp_hr)[1] + sum(tmp_hr$qt_st!='<null>')); # **Not used!**
      tmp_hr$hr <- j + 1;                                     # Track the hour.
      
      # ----- Populate dataframe ----
      ratio_date <- rbind(ratio_date, tmp_hr[1, c("in_st","sp_usr_id","in_qt_ratio","mod_ratio","hr","category_id","location_id")]); 
    }
  }
  
  ratio_date <- ratio_date[-1,];                                            # Remove top-row.
  ratio_date <- xts(ratio_date[,-1],order.by = as.POSIXct(ratio_date[,1])); # Order by Service provider user id.
  
  ep <- endpoints(ratio_date,'hours');
  tmp_ep <- period.apply(ratio_date[,2],ep,mean);             # Mean of qoute-to-invite ratios within the hour.
  
  # ---- Prepare data for Regression ----
  tmp_data <- tmp_ep[,1];
  tmp_data <- cbind(tmp_data,seq(1,dim(tmp_ep)[1]));
  colnames(tmp_data)[2] <- "hr";
  
  # ---- GAMLSS Regression ----
  reg_gam <- tryCatch(gamlss(in_qt_ratio ~ 0+hr, data=tmp_data,
                             family=BEINF(mu.link = "logit", sigma.link = "logit", nu.link = "log", tau.link = "log")), error=function(e) NULL );
  
  tmp_sum <- summary(reg_gam);          
  if(tmp_sum[1]==0){
    next;                                                     # If the optimizer did not converge then skip.
  }
  
  # ------ Populate dataframe of coefficients ---
  str_coef[k,1] <- ratio_date[1,5];
  str_coef[k,2] <- tmp_sum[1,1]; 
  str_coef[k,3] <- tmp_sum[1,2]; 
  str_coef[k,4] <- tmp_sum[1,4];
  str_coef[k,5] <- as.character(tmp_cat$place[1]);
  
}

# ----- Add Columns ---- #
# tmp_data <- data.frame(matrix(0, nrow = dim(dataAll)[1], ncol=4));
# colnames(tmp_data) <- c("in_qt_ratio","mod_ratio","hr","time");
# dataAll <- cbind(dataAll, tmp_data);


