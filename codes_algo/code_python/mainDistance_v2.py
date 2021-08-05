


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

def getString(strArray1,strArray2):
	
	tmpList = list(strArray1);
	tmpList.extend(list(strArray2));
	tmpList = sorted(set(tmpList));
	
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
	
	return (allLetter, strArray1, strArray2);

def sub_vowel(letter, cost, oriWord, edits):
	for i in letter:
		for j in letter:
			if (i.vowel != 0 and j.vowel != 0):
				if(i.isActive() < 0 and j.isActive() > 0):
					freq_i = i.getFreq()[0] - 1;
					freq_j = j.getFreq()[0] + 1;
					i.setFreqOne(freq_i);
					j.setFreqOne(freq_j);
					oriWord[i.pos[0]] = j.alpha;	
					edits.append((i.pos[0], j.alpha));
					i.pos.pop(0);
					cost = cost + 0.5;

	return (cost, oriWord, edits);

def sub_symbol(letter, cost, oriWord, edits):
	for i in letter:
		for j in letter:
			if(i.isActive() < 0 and j.isActive() > 0):
	 			freq_i = i.getFreq()[0] - 1;
				freq_j = j.getFreq()[0] + 1;
				i.setFreqOne(freq_i);
				j.setFreqOne(freq_j);	
				oriWord[i.pos[0]] = j.alpha;	
				edits.append((i.pos[0], j.alpha));
				i.pos.pop(0);
				cost = cost + 1;

	return(cost, oriWord, edits);

def del_symbol(letter, cost, oriWord, edits):
	rec_del = [];
	for i in letter:
		if (i.isActive != 0):
			while (i.getFreq()[0] > i.getFreq()[1]):
				i.setFreqOne(i.getFreq()[0] - 1);
				adj_del = sum(k < i.pos[0] for k in rec_del);
				del[oriWord[i.pos[0] - adj_del]];
				rec_del.append(i.pos[0]);
				edits.append((i.pos[0] - adj_del));
				i.pos.pop(0);
				cost = cost + 2;

	return (cost, oriWord, edits);

def ins_symbol(letter, cost, oriWord, tgtWord, edits):
	for i in range(len(tgtWord)):
		if (len(oriWord) == len(tgtWord)):
			break;
		elif (len(oriWord) == i):
			oriWord.insert(i,tgtWord[i]);	
			edits.append((oriWord[i], i));
			cost = cost + 3;
		else:		
			if (oriWord[i] != tgtWord[i]):
				oriWord.insert(i,tgtWord[i]);	
				edits.append((oriWord[i], i));
				cost = cost + 3;

	return(cost, oriWord, edits);

def distance(letter, oriWord, tgtWord):
	cost = 0; old_cost = 0.1;
	edits = [];
	while (cost != old_cost):
		old_cost = cost;
		[cost, oriWord, edits] = sub_vowel(letter, cost, oriWord, edits);
	
	old_cost = 0.1;	
	while (cost != old_cost):
		old_cost = cost;
		[cost, oriWord, edits] = sub_symbol(letter, cost, oriWord, edits);
	
	[cost, oriWord, edits] = del_symbol(letter, cost, oriWord, edits);

	[cost, oriWord, edits] = ins_symbol(letter, cost, oriWord, tgtWord, edits);

	return(cost, oriWord, edits);

def permComp():
	strArray1 = raw_input("Enter first letter: ");
	strArray2 = raw_input("Enter second letter: ");
	
	[uniqLetters, oriWord, tgtWord] = getString(strArray1,strArray2);
	actList = [ ];
	j = 0;
	for i in uniqLetters:
		if (i.isActive() != 0): 
			actList.append(i.alpha);
			j += 1;
		
	permutations = ["".join(x) for x in itertools.permutations(actList, len(actList))];

	dis = [ ];
  	finWord = list();	
	orgLetter = numpy.ndarray((len(actList),),  dtype = numpy.object);
	strOriWord = list(oriWord);
	chk_list = [];
	edits_list = [];
	j = 0;
	for i in permutations:
		n = 0;
		[uniqLetters, oriWord, tgtWord] = getString(strArray1,strArray2);
        	if(j == 1440):
			b = 4;
		for k in list(i):
			for m in uniqLetters: 
				if (k == m.alpha): 
					orgLetter[n] = letter();
					orgLetter[n] = m; 
					n +=1 ; 

		[tmp_dis, tmp_oriWord, tmp_edits] = distance (orgLetter, list(oriWord), list(tgtWord));
		if (tmp_oriWord == list(tgtWord)):
			dis.append(tmp_dis);
			finWord.append(tmp_oriWord);
			if (sorted(tmp_edits) in chk_list):
				continue;
			else:
				chk_list.append(sorted(tmp_edits));
				edits_list.append([tmp_dis,tmp_edits]);

		oriWord = list(strOriWord);
		j += 1;
 
	return(edits_list)

def main(argv=None):

	edits  = permComp();
	for i in edits:
		print("Distance = " + str(i[0]));		
		for j in i[1]:
			if (type(j) != type(0)):	
				if(type(j[0]) == type(0)):
					print("replace character from position " + str(j[0]) + " by " + j[1]);
					a = 9;	
				else:
					print("insert character " + j[0] + " at position " + str(j[1]));
			else:
				print("delete character in position " + str(j));
			
	return None;

if __name__ == '__main__':
	status = main();
	sys.exit(status);	

