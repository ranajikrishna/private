

import sys;
import numpy;
from random import*
import itertools;
from math import*
from decimal import*

def main(argv=None):
	arraySum = numpy.zeros(100);
	for j in range(0,99):
		tot = 0; 
		i = 1; 
		while tot < 100:
			tmp = randint(1,10);
			tot += tmp;  
			i += 1;
		
		arraySum[j] = Decimal(tot)/Decimal(i-1);
		
	return(sum(arraySum)/100);

if __name__ == '__main__':
	status = main();
	sys.exit(status);

