

import sys;
sys.modules[__name__].__dict__.clear()	# Clear All.



 # ------------- String Manipulation ------------ #
x = "Hello World!"
y = " to someone"

#print x
#print(x+y)
#print (x * 2)
#
#print "Hi" + ', how are you?'
#print "\"Hi\" said the person"
#print "He\'s gone"
#print "Hi\nwhat's your name?"
#print "hEllo\\"
#
#print "hello\tworld"
#print "hello\vworld"
#
#print (4**2)
#print "/*------ Ranaji Krishna ------- */"

# ------------------- #


# ------------- Lists in Python ------------ #

water = [2,4,3,5,7,5, "hi", 4];   # List
#print (water);
#print water[-4];
#print water[2:5];
#print water[-4:-1];
#print water[:];

tmp = [6,4, sorted(water), 'Ranaji', 7, 8, 3, 2, '45', 4, 'Krishna'];

#print (tmp);
#print (sorted(tmp));

tmp.append(100);   # Append an element into a List.
#print tmp;

tmp.extend(water); # Appends a list (water) into a list (tmp).
print tmp;

tmp.reverse(); 	   # Reverse a List.
#print tmp;

tmp = tmp[::-1];   # Same as reverse().
#print tmp;

tmp.pop(); 	   # Removes the last item in the list.
#print tmp;

#print tmp;
tmp.pop(2);	   # Removes the 3rd element in the list. 
#print tmp;

#print tmp.index("Krishna"); # Prints the index of an element of a List.

tmp.insert(4, "Dev");	     # Insert an element into a List.
#print tmp;

# ---------------- #


# --------- Input and Raw Input ---------- #

#tmpIn = input();				 # Input of strings require "".
#print tmpIn;

#firstName = input("Enter your name: ");	 # Input takes inputs as int.	
#print firstName;

#lastName = raw_input("Enter your last name: "); # Input takes input as string.
#print lastName;				 	

#print "Hello" + " " + firstName + " " + lastName; 

# --------------- #


# --------- Type Conversion --------- #

a = 72;

#print a;
#print (str(a));
#print float (a);
#print (list(str(float(a))));	# List.
#print (tuple(str(float(a))));	# Tupple.

b = `a`;
#print (b);			# To convert into string.

# -------------- #


# --------- Conditional - if -------- #

#num = input ("Enter a number:");
#num = int (num);
#
#if num > 10:
#	print "The number is greater than 10.";
#if num % 2 == 0:
#	print "The number is Even.";
#
#if num % 2 == 0 and num > 100:
#	print "The number is Even and greater than 100.";
#
#if num % 2 == 0 or num > 100:
#	print "The number is either Even or greater than 100 or both."

# --------------- #


# --------- Conditionals - Else and Elseif ---- #

#num = input("Enter a number: ");
#num = int (num);
#
#if num < 100:
#	print "The number is less than 100.";
#elif num >= 100 and num <= 1000:	
#	print "The number is between 100 and 1000.";
#else:	
#	print "The number os more than 1000.";

#a = raw_input ("Enter a string sequence: ");
#if "k" or " " in a:
#	print "Letter \"k\" is in the sequence or there is a space in it.";
#else: 	print "Letter \"k\" is not in the sequence.";

# -------------- #

# --------- Conditionals - Nested Conditionals ---- #

#a = "hello";
#print (a == "hello");
#print (a is "hello");
#print (a != "hello");
#print (a is not "hello");

#num = raw_input ("Enter a number: ");
#num = int (num);
#
#if num < 0:
#	print "The number is negative.";
#else:
#	if num > 100:
#		print "The number is gretaer than 100";
#	else:
#		print "The number is less than 100";

# --------------- #

# ---------- Type Checking -------- #

#a = "42"
#b = 42;
#c = 42.0;
#d = [42];
#e = (42);
#f = 42 , 42;

#print (type (a));
#print (type (b));
#print (type (c));
#print (type (d));
#print (type (e));
#print (type (`c`));
#print (type (f));

# -------------- #


# ---------- Warnngs with Floats --------- #

#from math import *
#
#print (sqrt (2));
#
#if (sqrt(2) * sqrt(2) == float(2.0)) : 	   # It equals 2.0000000000000000004!! 
#	print 2;
#else: print 3
#
#print (type(sqrt(2)*sqrt(2)))
#
#print (abs (-9));

# ------------ #

# ----------- While Loops --------- #

#a = 10
#b = 0;
#while a > b :
#	print(a - b);
#	a -= 1;
#
#print (a);

# ----------- #

# ----------- For Loops ----------- #

#print(range(10));
#print(range(0,10,2));
#print(range(15,10,-1));
#
#for m in range (10):
#	print m;

# ----------- #

# ----- Looping thorugh Sequences ----- #

#a = "hello world";
#b = tmp;
#c = 42,45,"34",24,"hello34523";
#
#for itr in c:					# Replace 'a', 'b' and 'c'.
#	print itr;

# ------------ #

# ----- Break Statement ----- #
#a = 1;
#while a!= 4: 
#	b = raw_input("Enter the result of 5 + 5: ");
#	a += 1;
#	print (type(b));
#	if b == `10`:				# Note: "b" is a string.
#		break

# ----------- # 


# ----- Setes ------ #
#jimmylai = ['abc', 'def', 'ghi', 'jkl'];
#print(jimmylai);
#
#jimmy = set(jimmylai);
#print (jimmy);
#
#print('def' in jimmylai);
#print('def' in jimmy);
#
#a = set ("Hello World");
#b = set ("Hello");
#print (a);
#print (b);
#print (type(a));
#print (a - b);
#print (a | b);			# Prints all letters unique to a and b.
#print (a & b);			# Prints all letter comoon to a and b.
#
#print('def' in jimmy);

# -------- #


# ------------- Dictionaries ----------- #

# Dictionaries use Hash Tables. Therefore they are faster in retriving data 
# than Lists. Dictionaries use key-value pairs. Dictionaries are un ordered, 
# hence index numbers don't work; we use the keys instead.


#ages = {("liam"):42, "timothy":4, "evan":5};
#print (ages["liam"]);
#print ("timothy" in ages);
#
#a = {('h','e','l','l','o','1','2','3'):"hello321", 1:1, 0.01:None}
##print (a);
#print (a[1]);	

# ----------------- #

# ----------- Importing Modules ------- #
#import math;
#print(math.sqrt(16));
##print (log (16)); 		# Will *not* work.
#print(math.log(16));		# Will work.

#from math import log;		# Puts the fxn. Log into the user namespace.
#from math import sqrt;
#print(log(10));		# WIll work.		
#print(sqrt(16));

#from math import *		# This will import all fxn. from math into the usernamespace.

#import random as test_ran;
##print (randint(0,10));        # Will not work coz 'randint' is a fxn of module 'random'...
#print(test_ran.randint(0,10));

#from random import *
#print(randint(0,10));		# ...but this will work.	

##help (math)			# Provides man page for all fxns. in math module.

# ----------------- #



# -------- Fucntions - The Basics --- #

#def function_name(para1, para2):
#	body_of_function
#       return

# ----
#def sayHi():
#	print ("Hello!");

#a = sayHi();
#print(sayHi());
#print (a);
# ----
# ----
#def sayHi():
#	return ("Hi")
#
#a = sayHi();
#print(sayHi());
#print (a);
# ---

#def makeFloat(x):
#	x = float(x);
#	return (x);
#
#print(makeFloat(5));
#print (x);			# x is not a global variable.	
# ----------------- #

# -------- Object Oriented Programming ----- #

#class exampleClass:
#	eyes = "blue";
#	age = 22;
#	def thisMethod (self):
#		return "hey this method worked";

#print (exampleClass.eyes);
#print (exampleClass.age);

#exampleObject = exampleClass();  
#print (exampleObject.thisMethod());
#print (exampleObject.eyes);

# ----------- #

# --------- Classes and Self ------ #

class className ():
	def createName (self,name):
		self.myName = name;
       
	def displayName (self):
		return self.myName;
       
	def saying  (self):
		print "hello %s" % self.myName;

	def test (self):
		self.testName = self.myName;
		print (self.testName);

first = className();
second = className();

first.createName("Ranaji");
second.createName("Vini");

print(first.displayName());
print(first.saying());
print(first.test());

#self is a tmeporary place holder for the object's name. In the first
#case the self is replaced by first (the object name).
#In the second case it is replaced by second. 

# ---------------- #

# --------- Subclasses and Superclasses ---- #

#class parentClass():
	
#	var1 = "I'm var 1";
#	var2 = "I'm var 2";

#class childClass(parentClass):       	     # The childClass Inherits the parentClass.
#	pass;


#parentObject = parentClass();
#childObject = childClass();
#print(parentObject.var1);
#print(childObject.var2)
	
# --------------- #

# --------- Overwritting Variable on Sub --- #

#class parent:
#	var1 = "bacon";
#	var2 = "sausage";

#class child(parent):
#	var2 = 'toast';

#pob = parent();
#cob = child();

#print(pob.var2);
#print(cob.var2);

# -------------- #

# --------- Inheriting Multiple Classes ----- #
#class Mom:
#	var1 = "Hey I'm Mom";

#class Dad:
#	var2 = "Hey I'm Dad";

#class Child (Mom, Dad):
#	var3 = "Hey I'm new";

#childObject = Child();
#print(childObject.var1);
#print(childObject.var2);
#print(childObject.var3);

# ------------- #


# -------- Constructor --------- #

#class new:
#	def __init__(self,name):
#		print ("This is a constructor");
#		print ("This also prints out");
#		self.myName = name; 
#
#newObj = new("Ranaji");
#print(newObj.myName);



