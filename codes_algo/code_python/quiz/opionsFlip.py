

import sys;
import numpy;
from random import*
import itertools;
from math import*
from decimal import*

def main(argv = None):
	strArray1 = raw_input("Enter Word:");
	strArray2 = strArray1.replace(" ","").replace(",","").replace("!","").replace(":","").replace(".","").replace("-","");
	strArray = list(strArray2.lower());
	i = 0;
	j = 0;
	while(i <= int(0.5*(len(strArray) - 1))):
		if(strArray[i] != strArray[int(len(strArray)) - i - 1]):
			j = 1;
			print("Not a Palindrome");
			break;
		else:
			i += 1;
	
	if(j != 1):	
		print("Is a Palindrome");


if __name__ == '__main__':
	status = main();
	sys.exit(status);

