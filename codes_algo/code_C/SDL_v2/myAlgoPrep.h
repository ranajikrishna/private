
#include <stdio.h>
#include <stdlib.h>

#define MAX_CHARS 200
typedef char string[MAX_CHARS+1];  
struct uniqWord{
	string word[200];
	int freq;
};

struct uniqWord *quickSortStruct(struct uniqWord *pArray, int strt, int end);

