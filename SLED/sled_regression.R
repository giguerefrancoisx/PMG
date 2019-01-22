setwd('C:/Users/tangk/Python/')
source("read_data.R")
source("helper.R")
library(lme4)
library(jsonlite)

directory <- 'P:/Data Analysis/Projects/SLED'

table <- read_table(directory,
                    filt=c('SE','MODEL','INSTALL','DUMMY','SLED'),
                    query=c('DUMMY == \'Y2\'',
                            'INSTALL %in% c(\'B11\',\'B12\',\'C1\',\'B0\')',
                            'SLED %in% c(\'new_accel\',\'new_decel\')'),factor=FALSE)
rownames(table) <- table$SE

features <- read.csv(file.path(directory,'features.csv'))
rownames(features) <- features$X

features_subset <- features[rownames(table),]
features_subset <- features_subset[!rownames(features_subset) %in% c('SE16-0253','SE16-0257','SE16-0351','SE15-1278'),]


# multiple linear regression
model <- lm(Chest_3ms ~ Min_DDown_x + TDDown_y.Angle, data=features_subset)
summary(model)

# do a regression for all samples with TDDown_y.Angle > 0


plot(model$fitted.values,col='red')
par(new=TRUE)
plot(features_subset$Chest_3ms,col='green')



model2 <- lm(Chest_3ms ~ Max_Angle + TDDown_y.Angle, data=features_subset)
summary(model2)

res <- data.frame(pred=model$fitted.values,act=features_subset$Chest_3ms)
write.csv(res, file.path(directory,'predictions.csv'))
