
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
setwd('/Users/vashishtha/myGitCode/hipMunk/')                       # Set directory
#dataAll <- read.csv('delay_data.csv', stringsAsFactor = TRUE);     # Import data.
#dataAll <- dataAll[,!(names(dataAll) %in% c("X"))]
#save(dataAll,file="dataAll.Rda");
load(file="dataAll.Rda")


cancels <- tapply(dataAll$CANCELLED, dataAll$FL_NUM, mean)
diverts <- tapply(dataAll$DIVERTED, dataAll$FL_NUM, mean)
dataAll <- cbind(dataAll, strftime(as.POSIXct(str_pad(dataAll$DEP_TIME, 4, pad="0"), format="%H%M") - (dataAll$DEP_DELAY * 60),format="%H:%M"),
                          strftime(as.POSIXct(str_pad(dataAll$ARR_TIME, 4, pad="0"), format="%H%M") - (dataAll$ARR_DELAY * 60),format="%H:%M"),
                          apply(as.array(dataAll$FL_NUM), 1, function(x) as.numeric(cancels[x])),
                          apply(as.array(dataAll$FL_NUM), 1, function(x) as.numeric(diverts[x])),
                          dataAll$DEP_DELAY + dataAll$TAXI_OUT + dataAll$ACTUAL_ELAPSED_TIME + dataAll$TAXI_IN)

colnames(dataAll)[16] <- "ADV_DEP_TIME"
colnames(dataAll)[17] <- "ADV_ARR_TIME"
colnames(dataAll)[18] <- "JRY_TIME"

data_colNames<- colnames(dataAll)
dataAll <- dataAll[,c(data_colNames[1:6],
                      data_colNames[16],
                      data_colNames[7:10],
                      data_colNames[17], 
                      data_colNames[11:15], 
                      data_colNames[18])]

# ori <- "LAX"  #"JFK"
# des <- "SFO"  #"DCA"

sub_data <- subset(dataAll, dataAll$ORIGIN==ori & dataAll$DEST==des)
#sub_data <- subset(dataAll[complete.cases(dataAll),], dataAll[complete.cases(dataAll),"ORIGIN"]==ori & dataAll[complete.cases(dataAll),"DEST"]==des)
sub_data <- sub_data[order(sub_data$ADV_ARR_TIME),]

flight <- list(ADV_DEP_TIME = tapply(as.character(sub_data$ADV_DEP_TIME), as.character(sub_data$ADV_ARR_TIME), function(x) unique(x[!is.na(x)])), 
               FLIGHT_NUM = tapply(sub_data$FL_NUM, as.character(sub_data$ADV_ARR_TIME), function(x) median(x, na.rm = TRUE)),
               DIVERSION = tapply(sub_data$DIVERTED, as.character(sub_data$ADV_ARR_TIME), function(x) mean(x, na.rm = TRUE)),
               DEP_AVG_DELAY = tapply(sub_data$DEP_DELAY, as.character(sub_data$ADV_ARR_TIME), function(x) median(x, na.rm = TRUE)),
               AVG_TAXI_OUT = tapply(sub_data$TAXI_OUT, as.character(sub_data$ADV_ARR_TIME), function(x) median(x, na.rm = TRUE)),
               ARR_AVG_DELAY = tapply(sub_data$ARR_DELAY, as.character(sub_data$ADV_ARR_TIME), function(x) median(x, na.rm = TRUE)),
               AVG_TAXI_IN = tapply(sub_data$TAXI_IN, as.character(sub_data$ADV_ARR_TIME), function(x) median(x, na.rm = TRUE)), 
               AVG_ELASPED = tapply(sub_data$ACTUAL_ELAPSED_TIME, as.character(sub_data$ADV_ARR_TIME), function(x) mean(x, na.rm = TRUE)),
               AVG_JRY = tapply(sub_data$JRY_TIME, as.character(sub_data$ADV_ARR_TIME), function(x) mean(x, na.rm = TRUE)),
               FLIGHT_DATES = tapply(sub_data$FL_DATE, as.character(sub_data$ADV_ARR_TIME), function(x) format(as.Date(as.character(x)),"%d"))
              )


# flight <- list(ADV_DEP_TIME = tapply(as.character(sub_data$ADV_DEP_TIME), as.character(sub_data$ADV_ARR_TIME), unique), 
#                FLIGHT_NUM = tapply(sub_data$FL_NUM, as.character(sub_data$ADV_ARR_TIME), median),
#                DIVERSION = tapply(sub_data$DIVERTED, as.character(sub_data$ADV_ARR_TIME), mean),
#                DEP_AVG_DELAY = tapply(sub_data$DEP_DELAY, as.character(sub_data$ADV_ARR_TIME), median),
#                AVG_TAXI_OUT = tapply(sub_data$TAXI_OUT, as.character(sub_data$ADV_ARR_TIME), median),
#                ARR_AVG_DELAY = tapply(sub_data$ARR_DELAY, as.character(sub_data$ADV_ARR_TIME), median),
#                AVG_TAXI_IN = tapply(sub_data$TAXI_IN, as.character(sub_data$ADV_ARR_TIME), median), 
#                AVG_ELASPED = tapply(sub_data$TAXI_IN, as.character(sub_data$ADV_ARR_TIME), mean),
#                AVG_JRY = tapply(sub_data$JRY_TIME, as.character(sub_data$ADV_ARR_TIME), mean),
#                FLIGHT_DATES = tapply(sub_data$FL_DATE, as.character(sub_data$ADV_ARR_TIME), function(x) format(as.Date(as.character(x)),"%d"))
#                )

str_name <- names(flight)
flight <- lapply(seq(1,length(flight)), function(x) flight[[x]][sort.list(flight$AVG_JRY)])
names(flight) <- str_name
row.names(flight) <- "ADV_DEP_TIME"

View(flight)