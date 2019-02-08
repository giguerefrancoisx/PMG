# collection of statistical tests
library(boot)

mean2 <- function(x,i){
	m <- mean(x[i])
	se2 <- (sd(x[i])/sqrt(length(x[i])))^2
	return(c(m,se2))
	}

#dmean <- function(x,i){
#	}

#bootstrap.p <- function(x,y,N){
#	n <- length(x)
#	m <- length(y)
#	bs.sample <- boot(c(x,y),statistic=,N)
#	}

#bootstrap <- function(x,y,N=100){
#	# get bootstrap samples
#	x.bs <- boot(x,statistic=mean2,N)
#	y.bs <- boot(y,statistic=mean2,N)
#	cix <- boot.ci(x.bs)$student[4:5]
#	ciy <- boot.ci(y.bs)$student[4:5]
#	p <- 
#
#	return(list(mux=mean(x),muy=mean(y),sdx=sd(x),sdy=sd(y),bsmux=mean(x.bs$t),bsmuy=mean(y.bs$t),bscix=cix,bsciy=ciy,p=p))
#}

range <- function(x,y){
	# assume max(y)>=max(x)

	mx <- min(x,na.rm=TRUE)
	Mx <- max(x,na.rm=TRUE)
	my <- min(y,na.rm=TRUE)
	My <- max(y,na.rm=TRUE)
	
	if(mx>=my){return(NA)}
	else if(mx<my & Mx>my){return(-(Mx-my))}
	else if(mx<my & Mx<=my){return(my-Mx)}
	}

two.sample.test <- function(x,y,test,paired=FALSE,args=NULL){
  # do a two sample test on x and y 
  
	if(all(is.na(x)) | all(is.na(y))){return(c(NULL,NULL))}
	else if((sum(!is.na(x))<=1 | sum(!is.na(y))<=1) & test!="range"){return(c(NULL,NULL))}	


	# test is one of: ttest, bootstrap, wilcoxon, permutation, range
	if(test=="ttest"){
		res <- do.call(t.test,c(list(x,y=y,paired=paired,conf.int=TRUE),args))
		return(c(res$method,res$p.value,res$conf.int[1],res$conf.int[2],attr(res$conf.int,'conf.level')))
		}
	else if(test=="bootstrap"){return(c(bootstrap(x,y,N=N),"bootstrap"))}
	else if(test=="wilcoxon"|test=="wilcox"){
		res <- do.call(wilcox.test,c(list(x,y=y,paired=paired,conf.int=TRUE),args))
		return(c(res$method,res$p.value,res$conf.int[1],res$conf.int[2],attr(res$conf.int,'conf.level')))
		}
	else if(test=="permutation"){return(c(NULL,"permutation"))}
	else if(test=="range"){
		if(max(y,na.rm=TRUE)>=max(x,na.rm=TRUE)){return(c(range(x,y),"range"))}
		else{return(c(range(y,x),"range"))}
		}
	}

two.sample.byname <- function(x,y,test,paired=FALSE,args=NULL,vector.form=TRUE){
  # do two sample test on specified by test
  # samples are not time series 
  # returns results of test as well as mean and variance measure
  # res has one row
  
	nx <- dim(x)[1]
	ny <- dim(y)[1]
	
	res <- apply(rbind(x,y),2,function(j) two.sample.test(j[1:nx],j[(nx+1):(nx+ny)],test,paired=paired,args=args))

	testname <- as.character(res[1,1])
	alpha <- as.numeric(as.character(res[5,1]))
	res <- res[-c(1,5),]

	# calculate means and sds
	if(paired){
		means <- colMeans(x-y)
		sd <- apply(x-y,2,sd,na.rm=TRUE)
	} else{
		means <- colMeans(x)-colMeans(y)
		sd <- sqrt(apply(x,1,var,na.rm=TRUE)/nx + apply(y,1,var,na.rm=TRUE)/ny)
	}
	res <- rbind(res,means)
	res <- rbind(res,sd)

	if(vector.form){
		res <- as.vector(res)
		n <- c("p","lb","ub","mean","sd")
		names(res) <- rep(n,length(res)/length(n))
	}
	return(list(stats=res,testname=testname,alpha=alpha))
	}

two.sample.ts <- function(x,y,test,paired=FALSE,args=NULL){
  # do a two sample test on a time series 
	if (is.null(names(x))){x <- t(x)}
	if (is.null(names(y))){y <- t(y)}
	nx <- dim(x)[2]
	ny <- dim(y)[2]
	
	# use apply to test all times at once
	res <- data.frame(t(apply(cbind(x,y),1,(function(j) two.sample.test(j[1:nx],j[(nx+1):(nx+ny)],test,paired=paired,args=args)))))
	
	testname <- as.character(res[1,1])
	alpha <- as.numeric(as.character(res[1,5]))	
	
	res <- res[,-c(1,5)]
	names(res) <- c("p","lb","ub")

	if(paired){
		means <- rowMeans(x-y)
		sd <- apply(x-y,1,sd,na.rm=TRUE)
	} else{
		means <- rowMeans(x)-rowMeans(y)
		sd <- sqrt(apply(x,1,var,na.rm=TRUE)/nx + apply(y,1,var,na.rm=TRUE)/ny)
	}
	res$mean <- means
	res$sd <- sd
	return(list(stats=res,testname=testname,alpha=alpha))
	}