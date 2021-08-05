
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
require("Stringi")
require("stringr")
library("plyr")
library("MASS")

# ====== Data ===== #
setwd('/Users/vashishtha/myGitCode/hipMunk/')                       # Set directory
# dataAll <- read.csv('delay_data.csv', stringsAsFactor = TRUE)     # Import data.
# dataAll <- cbind(dataAll,paste(dataAll$CARRIER, dataAll$FL_NUM))  # Concatenate Flight Number.
# colnames(dataAll)[17] <- "CAR_FL"
# dataAll <- dataAll[,!(names(dataAll) %in% c("X","FL_NUM","CARRIER"))]   # Drop some unnecessary Columns.
# save(dataAll,file="dataAll.Rda")                                        # Save Data.
load(file="dataAll.Rda")                                                  # Load Data.

ori <- "LAX"
des <- "DAS"

sub_data <- subset(dataAll, dataAll$ORIGIN==ori & dataAll$DEST==des)      # Subset containing revelvant data.
sub_data <- sub_data[complete.cases(sub_data), ]


# ---- Compute Advertised times and assign percentage cancellations to flights ---
sub_data <- cbind(sub_data, strftime(as.POSIXct(str_pad(sub_data$DEP_TIME, 4, pad="0"), format="%H%M") - (sub_data$DEP_DELAY * 60),format="%H%M"),
                            strftime(as.POSIXct(str_pad(sub_data$ARR_TIME, 4, pad="0"), format="%H%M") - (sub_data$ARR_DELAY * 60),format="%H%M"),
                  ifelse(sub_data$DEP_DELAY < 15, 0, 1)
)                  

colnames(sub_data)[15] <- "ADV_DEP_TIME"
colnames(sub_data)[16] <- "ADV_ARR_TIME"
colnames(sub_data)[17] <- "BIN_DELAY"

k=4
sub_data$ADV_DEP_TIME <- format(strptime("1970-01-01", "%Y-%m-%d", tz="UTC") + 
                         round(as.numeric(as.POSIXct(str_pad(sub_data$DEP_TIME-700, 4, pad="0"), format="%H%M"))/(900*k))*(900*k),"%H%M")

bin_model <-model.matrix(~ADV_DEP_TIME + BIN_DELAY, data = sub_data)[,-1]

regResults <- glm(formula = BIN_DELAY ~.,family=binomial,data = data.frame(bin_model))
#regResults <- glm.nb(formula = BIN_DELAY ~.,link=log, data = data.frame(bin_model))

toselect.x <- summary(regResults)$coeff[-1,4] < 0.05
relevant.x <- names(toselect.x)[toselect.x == TRUE] 
sig.formula <- as.formula(paste("BIN_DELAY ~", paste(relevant.x, collapse = "+")))
sig.model <- glm(formula = sig.formula, data = data.frame(bin_model))


browser()

# --- Round times to the hour --- 
# dataAll$ADV_DEP_TIME <- round_any(as.numeric(levels(dataAll$ADV_DEP_TIME))[dataAll$ADV_DEP_TIME], 100, f=floor)
# dataAll$ADV_ARR_TIME <- round_any(as.numeric(levels(dataAll$ADV_ARR_TIME))[dataAll$ADV_ARR_TIME], 100, f=floor)

# sub_data$ADV_DEP_TIME <- round_any(as.numeric(levels(sub_data$ADV_DEP_TIME))[sub_data$ADV_DEP_TIME], 100, f=floor)
# sub_data$ADV_ARR_TIME <- round_any(as.numeric(levels(sub_data$ADV_ARR_TIME))[sub_data$ADV_ARR_TIME], 100, f=floor)


# ---- Compute Advertised times ---
# dataAll <- cbind(dataAll, strftime(as.POSIXct(str_pad(dataAll$DEP_TIME, 4, pad="0"), format="%H%M") - (dataAll$DEP_DELAY * 60),format="%H%M"),
#                   strftime(as.POSIXct(str_pad(dataAll$ARR_TIME, 4, pad="0"), format="%H%M") - (dataAll$ARR_DELAY * 60),format="%H%M"),
#                   ifelse(dataAll$DEP_DELAY< 15, 0, 1)
# )


