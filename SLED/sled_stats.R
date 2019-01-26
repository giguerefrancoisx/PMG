setwd('C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/SLED')
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/COM/read_data.R")
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/COM/helper.R")
library(lme4)
library(jsonlite)

directory <- 'P:/Data Analysis/Projects/SLED'
comparison <- c('bench','sled')
dummy <- c('Y7','Y2')

table <- read_table(directory,
                    filt=c('SE','MODEL','INSTALL','DUMMY','SLED'),
                    query=c('DUMMY %in% c(\'Y2\',\'Y7\')',
                            'INSTALL!=\'\'',
                            'INSTALL!=\'B22\'',
                            'INSTALL!=\'H2\''),factor=FALSE)
rownames(table) <- table$SE
stattable <- read.csv(file.path(directory,'features.csv'),stringsAsFactors = FALSE)
if("SE" %in% names(stattable)){
  rownames(stattable) <- stattable$SE
} else{
  rownames(stattable) <- stattable$X
}
responses <- names(stattable)[2:length(names(stattable))]
params <- fromJSON(file.path(directory,'params.json'))


res <- list()
for(d in dummy){
  install <- unique(table.query(table,c(paste0('DUMMY==\'',d,'\'')))$INSTALL)
  for(c in comparison){
    if(c=='bench'){
      exclude <- 'new_decel'
    } else {
      exclude <- 'old_accel'
    }
    
    model_variance <- data.frame()
    model_beta <- data.frame()
    pval <- data.frame()
    
    for(i in install){
      subset <- table.query(table,c(paste0('DUMMY==\'',d,'\''),
                                    paste0('INSTALL==\'',i,'\''),
                                    paste0('SLED != \'',exclude,'\'')))
      
      if(length(unique(subset$SLED))<=1){
        next
      }

      for(y in responses){
        if(d=='Y2' & grepl('Y7',y)){
          next
        } else if(d=='Y7' & grepl('Y2',y)){
          next
        }
        if(sum(!is.na(stattable[subset$SE,y]))<=1){
          next
        }
        
        data_in <- merge(subset[,c('MODEL','SLED')],stattable[subset$SE,y,drop=FALSE],by=0)
        
        loop_error <- tryCatch(mixed.model <- lmer(as.formula(paste0(y,'~ SLED + (1|MODEL)')),data=data_in[,],REML=FALSE),
                               error=function(e){
                                 print(paste(d,c,i,y))
                                 print(e)})
        
        if(inherits(loop_error,"simpleError")){
          print('Skipping...')
          next
        }
        
        mixed.model.null <- lmer(as.formula(paste0(y,'~ (1|MODEL)')),data=data_in[,],REML=FALSE)
        
        s <- summary(mixed.model)
        
        total_variance <- sum(as.data.frame(s$varcor)$vcov)
        beta <- s$coefficients[2,1]
        p_val <- anova(mixed.model.null,mixed.model)$`Pr(>Chisq)`[2]
        
        model_variance[i,y] <- total_variance
        model_beta[i,y] <- beta
        pval[i,y] <- p_val

      }
    }
    res[[d]][[c]][['var']] <- model_variance
    res[[d]][[c]][['beta']] <- model_beta
    res[[d]][[c]][['p']] <- pval
  }
}

params$res <- res
write_json(params,file.path(directory,'params.json'))