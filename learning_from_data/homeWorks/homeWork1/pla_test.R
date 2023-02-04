answer <- 0
probability <- 0
n <- 10
for(pp in 1:1000){
  x1<-runif(2,-1,1)
  x2<-runif(2,-1,1)
  slope<-(x1[2]-x2[2])/(x1[1]-x2[1])
  temp1 <- runif(n,-1,1)
  temp2 <- runif(n,-1,1)
  data <- cbind(temp1,temp2) #dataset containing coordinates
  y <- vector(length=n)
  c <- x1[2] - slope*x1[1] #constant in the line equation
  for(i in 1 : n) {
    ifelse(data[i,2]-slope*data[i,1]-c < 0,y[i] <- -1,y[i] <- 1)
  }
#   browser()
  # Learning Starts Here
  #########################################
  
  data <- cbind(c(rep(1,n)),data)
  w <- c(rep(0,3))
  g <- vector(length=n)
  notDone <- T
  iteration <- 1
  while(notDone){
    for(j in 1:n){
      g[j] <- t(w) %*% data[j,]
    }
    if(all(sign(g)==sign(y))){
      ans <- iteration
      notDone <- F
    }
    else{
      for(k in 1:n){
        if(sign(g[k])!=sign(y[k])){
          w <- w + y[k]*data[k,]
          break
        }
      }
    }
    iteration <- iteration + 1
  }
  
  ##########################################
  #Probability P[f(x)!=g(x)]
  ##########################################
  
  prob1 <- runif(500,-1,1)
  prob2 <- runif(500,-1,1)
  probData <- cbind(prob1,prob2)
  y <- vector(length=500)
  g <- vector(length=500)
  for(i in 1:500){
    ifelse(probData[i,2]-slope*probData[i,1]-c < 0,y[i] <- -1,y[i] <- 1);
  }
  
  probData <- cbind(c(rep(1,n)),probData)
  for(j in 1:500){
    g[j] <- t(w) %*% probData[j,]
  }
  
  miss <- 0
  for(i in 1:500){
    if(sign(g[i])!=sign(y[i])){
      miss <- miss + 1
    }
  }
  
  ###########################################
  
  probability <- probability + miss/500
  answer <- answer+iteration
}

print(answer/1000)
print(probability/1000)