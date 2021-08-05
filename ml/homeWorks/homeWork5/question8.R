
nIter = 100;
nIn = 100;
nOut = 10000;
eta = 0.01;
eOut = rep(NA, nIter);
nEpochs = rep(NA, nIter);

set.seed(123);
for (i in (1:nIter)) {
  # Initialization of f(x) and of matching In-sample
  x1 = runif(min=-1,max=1,2);
  x2 = runif(min=-1,max=1,2);
  f = lm(x2 ~ x1);
  dataIn = data.frame(x0 = rep(1,nIn), x1 = runif(min=-1,max=1,nIn), x2 = runif(min=-1,max=1,nIn));
  XIn = as.matrix(dataIn);
  YIn = ifelse((dataIn$x2 > predict(f,newdata=dataIn)),1,-1);
  # Initialization of gradient descent
  w = c(0,0,0);
  nEpoch = 0;
browser();  
  # Gradient descent
  repeat {
    nEpoch = nEpoch +1;
    # Permutate In-sample and update weight vector with each data point in turn
    permuted.ind = sample(nIn, nIn, replace=FALSE);
    for (j in permuted.ind) {
      w0 = w;
      v = (-YIn[j] * XIn[j,]) / (1 + exp(YIn[j] * (XIn[j,] %*% w0)));
      w = w0 - eta * v;
    }
    # Stop gradient descent as soon as delta(w, w0) < 0.01
    if (sqrt(sum((w - w0)^2)) < 0.01) break()
  }
  # Save number of epoch cycles required to reach exit condition 
  nEpochs[i] = nEpoch;
  
  # Initalize Out-sample 
  outCoords = runif(min=-1,max=1,2*nOut);
  outX1.ind = sample(2*nOut,nOut);
  outX1 = outCoords[outX1.ind];
  outX2 = outCoords[-outX1.ind];
  dataOut = data.frame(x0 = rep(1,nOut), x1 = outX1, x2 = outX2);
  XOut = as.matrix(dataOut);
  YOut = ifelse((dataOut$x2 > predict(f,newdata=dataOut)),1,-1);
  # Compute and store out-sample error matching weight vector w obtained through gradient descent
  eOut[i] = sum(sapply(1 + sapply(-1 * YOut * (XOut %*% w), exp), log)) / nOut;
}
# E(Eout)
avgEOut = mean(eOut);
# E(NEpoch)
avgNEpoch = mean(nEpochs);