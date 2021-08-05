
#include <stdio.h>
#include <stdlib.h>
#include "myAlgoPrep.h"

int partition(struct uniqWord *tmpArray, int st, int ed){
	
	int pvt = (tmpArray + ed)->freq;
	int int1 = st;
	struct uniqWord n = *(tmpArray+ed);

	while(st < ed){
		if((tmpArray + st)->freq < pvt){
			struct uniqWord m = *(tmpArray+int1);
			*(tmpArray + int1) = *(tmpArray + st);
			*(tmpArray + st) = m;
			int1++;
		}
	st++;
	}
	
	*(tmpArray + ed) = *(tmpArray + int1);
	*(tmpArray + int1) = n;

	return(int1);
}


struct uniqWord *quickSortStruct(struct uniqWord *pArray, int strt, int end){

	if(strt < end){
		int int1;
		int1 = partition(pArray, strt, end);
		pArray = quickSortStruct(pArray, strt, int1 - 1);
		pArray = quickSortStruct(pArray, int1 + 1, end);
	}
	return(pArray);
}
