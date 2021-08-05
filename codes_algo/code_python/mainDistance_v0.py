

import sys;
import numpy;
import random;
import itertools;
from math import *		        # Math fxns.


class letter():

	vowel = 0;			# Indicates Vowel (1: Vowel).
	pos = None;			# Position of letters. If letter appears both in string1 & string2, store position of string1.
	alpha = None;			# The letter.

	def setFreqOne(self, freq):
		self.freq1 = freq;	# Freq. of letter in string1.

	def setFreqTwo(self, freq):
		self.freq2 = freq;	# Freq. of letter in string2.
	
	def isActive(self):
		return(self.freq2 - self.freq1);	# If freq. in string1 != freq. in string2.

	def getFreq(self):
		return([self.freq1,self.freq2]);

def getString(strArray1,strArray2):			# Get the statistics of letters.
	
	tmpList = list(strArray1);
	tmpList.extend(list(strArray2));
	tmpList = sorted(set(tmpList));			# Unique letters.
	
	allLetter = numpy.ndarray((len(tmpList),),  dtype = numpy.object);
	
	dtry1 = {x:strArray1.count(x) for x in strArray1};
	dtry2 = {x:strArray2.count(x) for x in strArray2};
	j = 0;
	for i in tmpList:
		allLetter[j] = letter();
		allLetter[j].alpha = i;
		if (i in ['a','e','i','o','u']):	# Check if letter is vowel. 
			allLetter[j].vowel = 1;		
		if (i in dtry1.keys()):
			allLetter[j].setFreqOne(dtry1[i]);					   # Store freq. in string1.
			allLetter[j].pos = [k for k, n in enumerate(strArray1) if n == i];     	   # Store multiple positions.
		else:
			allLetter[j].setFreqOne(0);						   # Set freq. = 0 if letter is not in string1.
		if (i in dtry2.keys()):
			allLetter[j].setFreqTwo(dtry2[i]);					   # Store freq. in string2.
			if (allLetter[j].pos == None):
				allLetter[j].pos = [k for k, n in enumerate(strArray2) if n == i]; # Store position if letter appears in string2 only.
		else:
			allLetter[j].setFreqTwo(0);						   # Set freq- = 0 if letter is not in string2.
		j += 1;
	
	return (allLetter, strArray1, strArray2);

def sub_vowel(letter, cost, oriWord, edits):						# Replace vowels by vowels.
	for i in letter:
		for j in letter:
			if (i.vowel != 0 and j.vowel != 0):
				if(i.isActive() < 0 and j.isActive() > 0):
					freq_i = i.getFreq()[0] - 1;			# Replace letters.
					freq_j = j.getFreq()[0] + 1;
					i.setFreqOne(freq_i);				# Update freq.
					j.setFreqOne(freq_j);
					oriWord[i.pos[0]] = j.alpha;			
					edits.append((i.pos[0], j.alpha));		# Store values for printing the operations performed. 
					i.pos.pop(0);					# Position of remaining letter.
					cost = cost + 0.5;				# Update cost.

	return (cost, oriWord, edits);

def sub_symbol(letter, cost, oriWord, edits):						# Replace letters by letters.
	for i in letter:
		for j in letter:
			if(i.isActive() < 0 and j.isActive() > 0):
	 			freq_i = i.getFreq()[0] - 1;				# Replace letters.
				freq_j = j.getFreq()[0] + 1;
				i.setFreqOne(freq_i);					# Update freq.
				j.setFreqOne(freq_j);	
				oriWord[i.pos[0]] = j.alpha;	
				edits.append((i.pos[0], j.alpha));			# Store values for printing the operations performed.
				i.pos.pop(0);						# Position of remaining letter.
				cost = cost + 1;					# Update cost

	return(cost, oriWord, edits);

def del_symbol(letter, cost, oriWord, edits):						# Delete letters.
	rec_del = [];									# Record positions of letters deleted.
	for i in letter:
		if (i.isActive != 0):
			while (i.getFreq()[0] > i.getFreq()[1]):
				i.setFreqOne(i.getFreq()[0] - 1);
				adj_del = sum(k < i.pos[0] for k in rec_del);		# Adjust position of delete if position of previous delete is less than the positiono of current delete.
				del[oriWord[i.pos[0] - adj_del]];			# Delete letter.
				rec_del.append(i.pos[0]);				# Record the position of deleted letter.
				edits.append((i.pos[0] - adj_del));			# Store values for printing the operations performed.
				i.pos.pop(0);						# Position of remaning letter.
				cost = cost + 2;					# Update cost.

	return (cost, oriWord, edits);

def ins_symbol(letter, cost, oriWord, tgtWord, edits):					# Insert letters.
	for i in range(len(tgtWord)):
		if (len(oriWord) == len(tgtWord)):					
			break;								# *** Break if the length of modified word > target word [to reduce iterations].***
		elif (len(oriWord) == i):						# Insert letters at the end of the modified word.
			oriWord.insert(i,tgtWord[i]);					# Insert letter
			edits.append((oriWord[i], i));					# Store values for printing the operations performed.
			cost = cost + 3;						# Update cost.
		else:		
			if (oriWord[i] != tgtWord[i]):					# Insert letter at the start or in the middle of the modified word.
				oriWord.insert(i,tgtWord[i]);	
				edits.append((oriWord[i], i));
				cost = cost + 3;

	return(cost, oriWord, edits);

def distance(letter, oriWord, tgtWord):							# Compute the distance.
	cost = 0; old_cost = 0.1;
	edits = [];									# List to store the operations performed.
	while (cost != old_cost):
		old_cost = cost;
		[cost, oriWord, edits] = sub_vowel(letter, cost, oriWord, edits);	# Replace vowels by vowels.
	
	old_cost = 0.1;	
	while (cost != old_cost):
		old_cost = cost;
		[cost, oriWord, edits] = sub_symbol(letter, cost, oriWord, edits);	# Replace letters by letters
	
	[cost, oriWord, edits] = del_symbol(letter, cost, oriWord, edits);		# Delete letters.

	[cost, oriWord, edits] = ins_symbol(letter, cost, oriWord, tgtWord, edits);	# Insert Letters.

	return(cost, oriWord, edits);

def permComp():
	strArray1 = raw_input("Enter first letter: ");					# String1.
	strArray2 = raw_input("Enter second letter: ");					# String2.
	
	[uniqLetters, oriWord, tgtWord] = getString(strArray1,strArray2);		# Get unique letters.
	actList = [ ];									# *** Active List: letters that require operations to be performed [to reduce number of permutations] ***
	j = 0;
	for i in uniqLetters:								# Populate Active List.
		if (i.isActive() != 0): 
			actList.append(i.alpha);
			j += 1;
		
	permutations = ["".join(x) for x in itertools.permutations(actList, len(actList))];	# All permutations.

	orgLetter = numpy.ndarray((len(actList),),  dtype = numpy.object);			# Store letters organised as per the order in the permutation.
	strOriWord = list(oriWord);								# String1 as list.
	chk_list = [];										# Store the operations that have been performed before.
	edits_list = [];									# Store unique operations performed and the distance incurred.
	j = 0;
	for i in permutations:									# Iterae through permutations.
		n = 0;
		[uniqLetters, oriWord, tgtWord] = getString(strArray1,strArray2);		# Re-initialise uniqLetters.
		
		for k in list(i):								# Populate orgLetter.
			for m in uniqLetters: 
				if (k == m.alpha): 
					orgLetter[n] = letter();
					orgLetter[n] = m; 
					n +=1 ; 

		[tmp_dis, tmp_oriWord, tmp_edits] = distance (orgLetter, list(oriWord), list(tgtWord));		# Compute Distance (sum of costs) & operations performed.
		if (tmp_oriWord == list(tgtWord)):								# If the modified wprd = string2.
			if (sorted(tmp_edits) in chk_list):					# Check if the operation is unique.
				continue;
			else:
				chk_list.append(sorted(tmp_edits));				# Populate chk_list.
				edits_list.append([tmp_dis,tmp_edits]);				# Populate the edits_list.

		oriWord = list(strOriWord);							# Re-initialise to string1.
		j += 1;
 
	return(edits_list);

def main(argv=None):

	edits  = permComp();						# Get Distance and the Operations performed.
	for i in edits:
		print("Distance = " + str(i[0]));			# Print Distance.	
		for j in i[1]:
			if (type(j) != type(0)):			# Operation: *not* Delete.
				if(type(j[0]) == type(0)):		# Operation: Replace.
					print("replace character from position " + str(j[0]) + " by " + j[1]);
					a = 9;	
				else:					# Operation: Insert.
					print("insert character " + j[0] + " at position " + str(j[1]));
			else:						# Operation: Delete
				print("delete character in position " + str(j));
			
	return None;

if __name__ == '__main__':
	status = main();
	sys.exit(status);	

