/**
* Given an array of integers nums and an integer target, return indices of the 
* two numbers such that they add up to target.
* You may assume that each input would have exactly one solution, and you may 
* not use the same element twice.
* You can return the answer in any order.
*/

#include <stdio.h>
#include <stdlib.h>
#include "myAlgoPrep.h"	

int SIZE = 4;

struct Pair{
	int *a;
	int *b;
};

struct Pair *check_pair(int *array, int target)
{
	int *sort_array;
	struct Pair sum, *s;
	s = &sum;
	int array_size = 4;
	int i = array_size;
	int j = i;
	
	sort_array  = bubble_sort(array, array_size);
	int diff = target - *(array+i);

	while (i>0) {
		if ((j<0 || *(array+j) < diff)){
			i--;
			j = i;
			diff = target - *(array+i);
		} else if (*(array+j) > diff){
			j--;
		} else {
			s->a = (array+j); 
			s->b = (array+i);
			return s;
		}

	}
	return NULL;
}

int main(int argc, char* argv[])
{
	int array[5] = {1,4,2,5,6};
	int target = 9;
	struct Pair *s;
	
	s = check_pair(array, target);

 //	*(s->a)
 //	*(s->b)


	return(0);
}
