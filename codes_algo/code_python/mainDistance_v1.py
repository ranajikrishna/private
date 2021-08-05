

import sys;
import numpy;
import random;
import itertools
from math import *		        # Math fxns.


class letter():

	vowel = 0;
	pos = None;	
	alpha = None;

	def setFreqOne(self, freq):
		self.freq1 = freq;

	def setFreqTwo(self, freq):
		self.freq2 = freq;
	
	def isActive(self):
		return(self.freq2 - self.freq1);

	def getFreq(self):
		return([self.freq1,self.freq2]);

def getString(tmpList, strArray1, strArray2):
	
	allLetter = numpy.ndarray((len(tmpList),),  dtype = numpy.object);
	
	dtry1 = {x:strArray1.count(x) for x in strArray1};
	dtry2 = {x:strArray2.count(x) for x in strArray2};
	j = 0;
	for i in tmpList:
		allLetter[j] = letter();
		allLetter[j].alpha = i;
		if (i in ['a','e','i','o','u']): 
			allLetter[j].vowel = 1;		
		if (i in dtry1.keys()):
			allLetter[j].setFreqOne(dtry1[i]);
			allLetter[j].pos = [k for k, n in enumerate(strArray1) if n == i];
		else:
			allLetter[j].setFreqOne(0);
		if (i in dtry2.keys()):
			allLetter[j].setFreqTwo(dtry2[i]);
			if (allLetter[j].pos == None):
				allLetter[j].pos = [k for k, n in enumerate(strArray2) if n == i];
		else:
			allLetter[j].setFreqTwo(0);
		j += 1;
	
	return (allLetter);

def sub_vowel(letter, cost, oriWord):
	for i in letter:
		for j in letter:
			if (i.vowel != 0 and j.vowel != 0):
				if(i.isActive() < 0 and j.isActive() > 0):
					freq_i = i.getFreq()[0] - 1;
					freq_j = j.getFreq()[0] + 1;
					i.setFreqOne(freq_i);
					j.setFreqOne(freq_j);
					oriWord[i.pos[0]] = j.alpha;	
					#print("replace character in position " + str(i.pos[0]) + " by " + j.alpha);
					i.pos.pop(0);
					cost = cost + 0.5;
	return (cost, oriWord);

def sub_symbol(letter, cost, oriWord):
	for i in letter:
		for j in letter:
			if(i.isActive() < 0 and j.isActive() > 0):
	 			freq_i = i.getFreq()[0] - 1;
				freq_j = j.getFreq()[0] + 1;
				i.setFreqOne(freq_i);
				j.setFreqOne(freq_j);	
				oriWord[i.pos[0]] = j.alpha;	
				#print("replace character in position " + str(i.pos[0]) + " by " + j.alpha);
				i.pos.pop(0);
				cost = cost + 1;
	return(cost, oriWord);

def del_symbol(letter, cost, oriWord):
	for i in letter:
		if (i.isActive != 0):
			while (i.getFreq()[0] > i.getFreq()[1]):
				i.setFreqOne(i.getFreq()[0] - 1);
				oriWord.delete(i.pos[0], i.alpha);
				#print("delete character in position " + str(i.pos[0]));
				i.pos.pop(0);
				cost = cost + 2;
	return (cost, oriWord);

def ins_symbol(letter, cost, oriWord, tgtWord,j):
	for i in range(len(tgtWord)):
		if (len(oriWord) == len(tgtWord)):
			break;
		elif (len(oriWord) == i):
			oriWord.insert(i,tgtWord[i]);	
			cost = cost + 3;
		else:		
			if (oriWord[i] != tgtWord[i]):
				oriWord.insert(i,tgtWord[i]);	
				#print("insert character " + i.alpha + " at position " + str(tgtWord.index(i.alpha)));				
				cost = cost + 3;
	return(cost,oriWord);

def distance(letter, oriWord, tgtWord, j):
	cost = 0; old_cost = 0.1;
	while (cost != old_cost):
		old_cost = cost;
		[cost, oriWord] = sub_vowel(letter, cost, oriWord);
	
	old_cost = 0.1;	
	while (cost != old_cost):
		old_cost = cost;
		[cost, oriWord] = sub_symbol(letter, cost, oriWord);

	[cost, oriWord] = del_symbol(letter, cost, oriWord);

	[cost, oriWord] = ins_symbol(letter, cost, oriWord, tgtWord,j);
	return(cost, oriWord);

def permComp(oriWord, tgtWord):
	tmpList = list(oriWord);
	tmpList.extend(list(tgtWord));
	tmpList = sorted(set(tmpList));
	permutations = ["".join(x) for x in itertools.permutations(tmpList, len(tmpList))];

	dis = [ ]; #numpy.zeros(len(permutations));
	
	uniqletters = letter();
	strOriWord = list(oriWord);
	j = 0;
	for i in permutations:
		uniqLetters = getString(i, oriWord, tgtWord);
        	[tmp_dis, tmp_oriWord] = distance (uniqLetters, oriWord, tgtWord, j);
		if (tmp_oriWord == tgtWord):
			dis.append(tmp_dis);
			finWord = list(tmp_oriWord);
			break;
		oriWord = list(strOriWord);
		j += 1; 
	return()

def main(argv=None):
	strArray1 = "kitten"    #raw_input("Enter first letter: ");
	strArray2 = "splitting" #raw_input("Enter second letter: ");

	tmp  = permComp(list(strArray1), list(strArray2));	
	print("Distance = " + str(dis));	
	return None;

if __name__ == '__main__':
	status = main();
	sys.exit(status);	

