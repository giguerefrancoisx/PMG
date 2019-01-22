# miscellaneous helper functions
table.query <- function(table,query){
	for(i in 1:length(query)){
		table <- table[eval(parse(text=paste("table$",query[i]))),]
		}
	return(table)
	}

names.to.se <- function(table,type.names){
	out <- list()
	for(n in names(type.names)){
		if ('SE' %in% names(table)){
			out[[n]] <- as.character(table.query(table,type.names[[n]])[['SE']])}
		else{
			out[[n]] <- as.character(table.query(table,type.names[[n]])[['TC']])}
		}
	return(out)
	}

param.data.to.list <- function(dataframe){
	x <- data.frame(diag(as.matrix(dataframe)))
	names(x) <- names(dataframe)
	return(x)
	}

clean_json <- function(params){
	if(length(params$data[[2]][[1]])==0){
		params$data <- param.data.to.list(params$data)
	}
	
	if(!is.null(params$channels)){
		params$channels <- sapply(params$channels,function(x) paste0('X',x),USE.NAMES=FALSE)
	}
	return(params)
}