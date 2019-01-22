library(sas7bdat)
library(rhdf5)

# default function read hdf5 files. If specified read method is csv, then the function calls read_merged_csv
read_merged <- function(directory,filename,format='h5'){
  if (format=='csv'){return(read_merged_csv(directory,filename))}
  
  path <- 'P:\\Data Analysis\\Data\\TC\\Tests.h5'
  full_data <- list()
  for (f in filename){
    paste('Reading file',f)
    name <- gsub('-','N',f)
    full_data[[f]] <- h5read(path,name)
    flush.console()
  }
  return(full_data)
  
}

# read data
read_merged_csv <- function(directory,filename){
  if(grepl('.csv',filename[1])){withExt=1}
  else{withExt=0}
  
  full_data = list()
  for (f in filename){
    name <- gsub("\\(SAI\\)\\.csv","",f)    
    print(paste('Reading file',name))
    if(withExt){
      full_data[[name]] <- read.csv(file=file.path(directory,f),header=TRUE)
    }
    else{
      full_data[[name]] <- read.csv(file=file.path(directory,paste(f,'(SAI).csv',sep='')),header=TRUE)
    }
    flush.console()
  }
  return(full_data)
}

read_dir <- function(directory){
	files <- list.files(directory,pattern=".csv")
	return(read_merged(directory,files))	
	}

read_merged_sas7bdat <- function(directory,filename){
  full_data <- list()
  for(f in filename){
    full_data[[gsub('\\.sas7bdat','',f)]] <- read.sas7bdat(file.path(directory,f))
  }
  return(full_data)
}


arrange_bych <- function(data,channels=NULL,cutoff=NULL){
  if (is.null(channels)){channels = colnames(data[[1]])} # fix this part later
  if (is.null(cutoff)){cutoff = 1:length(data[[1]][,1])} # get all times
  files = names(data)
  chdata = list()
  
  for(ch in channels){
    df <- data.frame(row.names=cutoff)
    flabel <- c()
    for(f in files){
      if(!is.null(data[[f]][[ch]][cutoff])){
        flabel <- append(flabel,f)
        df <- cbind(df,as.data.frame(data[[f]][[ch]][cutoff]))}
      else{next}
    }
    names(df) <- flabel
    chdata[[ch]] <- df
  }
  return(chdata)
}

read_table <- function(directory,filt=NULL,query=NULL,factor=FALSE){
	table <- read.csv(file=file.path(directory,'Table.csv'),header=TRUE,stringsAsFactors = factor)
	if(!is.null(filt)){
		table <- table[,filt]
		}
	if(!is.null(query)){
		for(i in 1:length(query)){
			table <- table[eval(parse(text=paste("table$",query[i]))),]
			}
		}
	return(table)
	}