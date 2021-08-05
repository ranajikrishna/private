
# -------- Invite-Quote trend analysis ----
# Author: Ranaji Krishna
# Date: July 13 2015.
# Notes: The code was developed to analyse trends across the entire data set.

# ---------- #


rm(list = ls(all = TRUE));  # clear all.
graphics.off();             # close all.

library("gdata");           # Importing excel data.
library("xts");
library("data.table");
library("zoo");
library("mondate");
library("grDevices");
library("scales");
library("ggplot2");
require("xlsx");
library("Matrix");

# ====== Data ===== #
setwd('/Users/vashishtha/myGitCode/hipMunk/')                       # Set directory
# dataAll <- read.csv('delay_data.csv', stringsAsFactor = TRUE)     # Import data.
# dataAll <- cbind(dataAll,paste(dataAll$CARRIER, dataAll$FL_NUM))  # Concatenate Flight Number.
# colnames(dataAll)[17] <- "CAR_FL"
# dataAll <- dataAll[,!(names(dataAll) %in% c("X","FL_NUM","CARRIER"))]   # Drop some unnecessary Columns.
# save(dataAll,file="dataAll.Rda")                                        # Save Data.
load(file="dataAll.Rda")                                                  # Load Data.

ori <- readline("Enter Origin: ")                                         # Prompt Origin.
des <- readline("Enter Destination: ")                                    # Prompt Destination.  

sub_data <- subset(dataAll, dataAll$ORIGIN==ori & dataAll$DEST==des)      # Subset containing revelvant data.

cancels <- tapply(sub_data$CANCELLED, as.character(sub_data$CAR_FL), mean)    # Percentage Cancellations. 

# ---- Compute Advertised times and assign percentage cancellations to flights ---
sub_data <- cbind(sub_data, strftime(as.POSIXct(str_pad(sub_data$DEP_TIME, 4, pad="0"), format="%H%M") - (sub_data$DEP_DELAY * 60),format="%H%M"),
                  strftime(as.POSIXct(str_pad(sub_data$ARR_TIME, 4, pad="0"), format="%H%M") - (sub_data$ARR_DELAY * 60),format="%H%M"),
                  apply(as.array(as.character(sub_data$CAR_FL)), 1, function(x) as.numeric(cancels[x]))
)

colnames(sub_data)[15] <- "ADV_DEP_TIME"
colnames(sub_data)[16] <- "ADV_ARR_TIME"
colnames(sub_data)[17] <- "CAN_PER"

# ----- Compute percentage DIversions and assign them to flights ---
diverts <- tapply(sub_data$DIVERTED, as.character(sub_data$ADV_DEP_TIME), mean)   
sub_data <- cbind(sub_data, apply(as.array(as.character(sub_data$ADV_DEP_TIME)), 1, function(x) as.numeric(diverts[x])));
colnames(sub_data)[18] <- "DIV_PER"

# ----- Compute Convinience measure for each flight -----
sub_data <- cbind(sub_data, (sub_data$DEP_DELAY + sub_data$TAXI_OUT + sub_data$ACTUAL_ELAPSED_TIME + sub_data$TAXI_IN)
                  * (0.0001 + sub_data$CAN_PER) * (0.0001 + sub_data$DIV_PER));
colnames(sub_data)[19] <- "CNV_MEA"

# ----- Re-order columns ----
data_colNames<- colnames(sub_data)
sub_data <- sub_data[,c(data_colNames[1:2], 
                        data_colNames[14], 
                        data_colNames[3:5],
                        data_colNames[15],
                        data_colNames[6:9],
                        data_colNames[16], 
                        data_colNames[10:11], 
                        data_colNames[17], 
                        data_colNames[12], 
                        data_colNames[18], 
                        data_colNames[13], 
                        data_colNames[19]
)]

sub_data <- sub_data[order(sub_data$ADV_DEP_TIME),]   # Order by Advertised Departure time.

# ----- Create List of flights with averages and medians -----
flight <- list(ADV_DEP_TIME = as.character(unique(sub_data$ADV_DEP_TIME)),
               ADV_ARR_TIME = tapply(as.character(sub_data$ADV_ARR_TIME), as.character(sub_data$ADV_DEP_TIME), function(x) unique(x[!is.na(x)])), 
               CARRIER_NUM = tapply(as.character(sub_data$CAR_FL), as.character(sub_data$ADV_DEP_TIME), function(x) unique(x[!is.na(x)])),
               CANCELLATION = tapply(sub_data$CAN_PER, as.character(sub_data$ADV_DEP_TIME), function(x) mean(x, na.rm = TRUE)),
               DIVERSION = tapply(sub_data$DIV_PER, as.character(sub_data$ADV_DEP_TIME), function(x) mean(x, na.rm = TRUE)),
               DEP_AVG_DELAY = tapply(sub_data$DEP_DELAY, as.character(sub_data$ADV_DEP_TIME), function(x) median(x, na.rm = TRUE)),
               AVG_TAXI_OUT = tapply(sub_data$TAXI_OUT, as.character(sub_data$ADV_DEP_TIME), function(x) median(x, na.rm = TRUE)),
               ARR_AVG_DELAY = tapply(sub_data$ARR_DELAY, as.character(sub_data$ADV_DEP_TIME), function(x) median(x, na.rm = TRUE)),
               AVG_TAXI_IN = tapply(sub_data$TAXI_IN, as.character(sub_data$ADV_DEP_TIME), function(x) median(x, na.rm = TRUE)), 
               AVG_ELASPED = tapply(sub_data$ACTUAL_ELAPSED_TIME, as.character(sub_data$ADV_DEP_TIME), function(x) mean(x, na.rm = TRUE)),
               AVG_CNV = tapply(sub_data$CNV_MEA, as.character(sub_data$ADV_DEP_TIME), function(x) mean(x, na.rm = TRUE)),
               FLIGHT_DATES = tapply(sub_data$FL_DATE, as.character(sub_data$ADV_DEP_TIME), function(x) format(as.Date(as.character(x)),"%d"))
)

str_name <- names(flight)
flight <- lapply(seq(1,length(flight)), function(x) flight[[x]][sort.list(flight$AVG_CNV)])   # Order by Convinience measure.
names(flight) <- str_name

View(flight)      # Display List of lists.

