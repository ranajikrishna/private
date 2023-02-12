
'''
Name: Gaussian Process simulation.
Author: Ranaji Krishna
Notes:
	y = f(x) is hypothesized to be a Gaussian process.
	This implies that a finite collection of y's (i.e. a vector
	of y) will have a multivariate Gaussian distribution. We 
	construct the multivariate distribution using x's (i.e. the vetor
	x (the independent variable)). Such a construction requires the
	mean vector and the covariace matrix (kernel) of the distribution
	to be designed. Both of these are functions of x, i.e. m(x) and C(x) 
	(and hold real values). This allows us to represent our views of the 
	relationship between x and y. For example, how far should two y's be
	when the x's are a certain distance apart. Once the mean and kernel 
	are designed, realisations of y can be drawn from the distribution 
	using y = Az + m, where A = chol(C) and z ~ N(0,I). We can draw many 
	sets (sample) of realisations of vector y. Here we show 10 sets (q = 10) 
	and 50 realistions in each set (i.e. y \mem R^{n X 1}). Note: 
	Each sample of y can be thought of as a fxn. because n can be taken to 
	infinity. 	
	Once the distribution has been constructed, it can be used for 
	regression analysis. This uses posterior distribution 
	p(y*|D) where y* is the dependent variable to be predicted and D is the 
	data {(x_i,y_i)} for every i = 1,...n. We have expressions for the mean
	and the variance which we use for evaluating the value of y* (the mean)
	and to establish the confidence interval (the variance).

'''

from my_quant_library import *


def kernel(a,b):
	sqdist = np.sum(a**2,1).reshape(-1,1) + \
	np.sum(b**2,1) - 2*np.dot(a,b.T)

	return np.exp(-.5 * sqdist)


def main (argv = None):

	n = 50	# Realisations.
	q = 10	# Sets (samples).
	Xtest = np.linspace(-5,5,n).reshape(-1,1)
	K_ = kernel(Xtest, Xtest)

	L = np.linalg.cholesky(K_ + 1e-8*np.eye(n))
	f_prior = np.dot(L, np.random.normal(size=(n,q)))


	plt.plot(Xtest, f_prior)

	return(0)


if __name__ == '__main__':
	status = main()
	sys.exit(status)
