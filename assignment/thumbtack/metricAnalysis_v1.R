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

# ====== Data =====
setwd('/Users/vashishtha/myGitCode/thumbTack/');   # Set directory
dataAll <- read.csv('universe.csv', stringsAsFactor = TRUE);         # Import data.

tmp_data <- data.frame(matrix(0, nrow = dim(dataAll)[1], ncol=3));
colnames(tmp_data) <- c("in_qt_ratio","mod_ratio","day");
dataAll <- cbind(dataAll, tmp_data);

ratio_date <- data.frame(matrix(0,1,ncol=7));
colnames(ratio_date) <- c("in_st","sp_usr_id","in_qt_ratio","mod_ratio","day","category_id","location_id");

day <- "2013-07-01";
for (i in 1 : max(dataAll$sp_usr_id) ){
  tmp_sp <- subset(dataAll, dataAll$sp_usr_id == i);
  for (j in unique(tmp_sp$in_st)){ 
    tmp_day <- subset(tmp_sp, tmp_sp$in_st == j);
    tmp_day$in_qt_ratio <- (sum(tmp_day$qt_st!='<null>')/dim(tmp_day)[1]);
    tmp_day$mod_ratio <- (dim(tmp_day)[1] - sum(tmp_day$qt_st!='<null>'))/(dim(tmp_day)[1] + sum(tmp_day$qt_st!='<null>'));
    tmp_day$day <- as.numeric(difftime(j,day),units="days") + 1;
    
    dataAll$in_qt_ratio[which(dataAll$sp_usr_id == i & dataAll$in_st==j)] <- (sum(tmp_day$qt_st!='<null>')/dim(tmp_day)[1])^1;    
    dataAll$mod_ratio[which(dataAll$sp_usr_id == i & dataAll$in_st==j)] <- (dim(tmp_day)[1] - sum(tmp_day$qt_st!='<null>'))/(dim(tmp_day)[1] + sum(tmp_day$qt_st!='<null>'));    
    dataAll$day <- as.numeric(difftime(j,day),units="days") + 1;
    
    ratio_date <- rbind(ratio_date, tmp_day[1, c("in_st","sp_usr_id","in_qt_ratio","mod_ratio","day","category_id","location_id")]); 
  }
}
ratio_date <- ratio_date[-1,];
ratio_date <- xts(ratio_date[,-1],order.by = as.POSIXct(ratio_date[,1]));

save(ratio_date,file="qt_in_ratio.Rda");
save(dataAll,file="dataAll_qt_in_ratio.Rda");
load(file="qt_in_ratio.Rda");
load(file="dataAll_qt_in_ratio.Rda");

# ratio_date[(is.infinite(ratio_date[,2])),2]<-10000;
# save(ratio_date,file="in_qt_ratio.Rda");
# save(dataAll,file="dataAll_in_qt_ratio.Rda");
# load(file="in_qt_ratio.Rda");
# load(file="dataAll_in_qt_ratio.Rda");

browser()

ep <- endpoints(ratio_date,'days');
tmp_ep <- period.apply(ratio_date[,3],ep,mean);

tmp_data <- tmp_ep[,1];
tmp_data <- cbind(tmp_data,seq(1,64));
colnames(tmp_data)[2] <- "day";

#reg_model_log <- glm(in_qt_ratio ~ day, data=tmp_data,family=binomial(link="logit"));
reg_model <- lm(tmp_ep[,1]~seq(1,64));
reg_mod_beta <- betareg(mod_ratio ~ day, data=tmp_data, link="loglog"); 
reg_mod_full <- betareg(mod_ratio ~ day, data=ratio_date, link="log"); 

pdf(file = ifelse(TRUE, "scat_fit.pdf"));           # Save plot.
plot(seq(1,64), tmp_ep[,1], main="Invite-to-Quote Ratio over Time", 
     xlab="Days", ylab="Ratio", pch=19);
lines(seq(1,64),reg_mod_beta$fitted.values,col="blue");
#abline(lm(tmp_ep~seq(1,64)), col="red");
dev.off();

pdf(file = ifelse(TRUE, "error_fit.pdf"));           # Save plot.
plot(seq(1,64), reg_model$residuals, main="Residual over Time", 
     xlab="Days", ylab="Residual", pch=19, col="blue");
abline(lm(tmp_ep~seq(1,64)), col="red");
dev.off();

shapiro.test(as.numeric(reg_model$residuals));
bptest(tmp_ep[,1]~seq(1,64));
bgtest(tmp_ep[,1]~seq(1,64));

dataAll_cat <- dataAll[with(dataAll,order(category_id,sp_usr_id)), ];
dataAll_loc <- dataAll[with(dataAll,order(location_id,sp_usr_id)), ];


