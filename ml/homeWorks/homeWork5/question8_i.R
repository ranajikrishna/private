#Q8
#we need a function for determining where a point falls
#given the 2 points p1 and p2 that determine the line

out <- function(p1,p2,p) {
  m <- (p2-p1)[2]/(p2-p1)[1]
  if (p[2] < m*(p[1]-p1[1])+p1[2]) return(-1) else return(+1)
}

#function for generating the training data. Size=N

generate_training_set <- function(p1,p2,N) {
  tm <- matrix(data=runif(min=-1,max=1,n=2*N),nrow=N,ncol=2)
  y <- sapply(1:N,function(i) out(p1,p2,tm[i,]))
  return( as.matrix(data.frame(x0=rep(1,N),x1=tm[,1],x2=tm[,2],y=y)) )
}

N <- 100
eta <- 0.01
runs <- 100
#the number of epochs and the Eout for each run
#will be stored in  the respective vectors
epochs <- numeric(0); Eouts <- numeric(0)
for (r in 1:runs) {
  #Generate the 2 points for the line
  p1 <- runif(min=-1,max=+1,n=2)
  p2 <- runif(min=-1,max=+1,n=2)
  #generate the training set of size N
  training_data <- generate_training_set(p1,p2,N)
  browser()
  w <- c(0,0,0); wp <- c(1,1,1); epoch <- 0
  while (sqrt(sum((w-wp)^2)) > 0.01) {
    wp <- w
    perm <- sample(1:N,size=N)
    for (j in 1:N) {
      i <- perm[j]
      grad <- (-training_data[i,4])*training_data[i,1:3]/
        (1+exp(training_data[i,4]*w%*%training_data[i,1:3]))
      w <- w - eta*grad
    }
    epoch <- epoch + 1
  }
  browser()
  epochs <- c(epochs,epoch)
  #Evaluate Eout, generate a new test set 10 times larger
  test_data <- generate_training_set(p1,p2,N*10)
  s <- sapply(1:(N*10),function(i) log(1+exp(test_data[i,1:3]%*%w*(-test_data[i,4]))) )
  Eout <- mean(s)
  Eouts <- c(Eouts,Eout)
  print(paste(r,epoch,Eout))    #so I can see what the program is doing
}