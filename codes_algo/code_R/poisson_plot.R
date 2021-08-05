

h = 10   # Rate
m = 20   # No of successes
plot(seq(1,m),exp(-h) * h ^{seq(1,m)}/apply(as.array(seq(1,m)),1,function(x) factorial(x)))


