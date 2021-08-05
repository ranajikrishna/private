

import numpy;

class myClass(object):
	def setFreq(self,freq1,freq2):
		self.freq1 = freq1;
		self.freq2 = freq2;

	def active(self):
		self.isActive = self.freq2 - self.freq1;
		return (self.isActive);
	
	def getFreq(self):
		return([self.freq1,self.freq2]);	

#def mulFxn(strOne, strTwo):
#	print(strOne.retFreq()[0] * strOne.retFreq()[0]);

if __name__ == '__main__':
	tmpClass1 = myClass();
	tmpClass1.setFreq(1,0);
	print(tmpClass1.getFreq()[1]);
	tmpClass2 = myClass();
	tmpClass2.setFreq(1,0);

	array = numpy.ndarray((10,), dtype = numpy.object);
	array[0] = myClass();
	array[0].setFreq(1,1);
	print(array[0].getFreq());

