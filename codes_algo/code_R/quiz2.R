

a <- matrix(0,6^3,3);
l <- 1
for(i in seq(1,6)){
  for(j in seq(1,6)){
    for(k in seq(1,6)){
      
      a[l,1] <- i;
      a[l,2] <- j;
      a[l,3] <- k;
      l <- l+1;
    }
  }
}