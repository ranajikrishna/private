
#include <stdio.h>
#include <stdlib.h>
#include "myAlgoPrep.h"	
			
int main(int argc, char* argv[]){

	int count = 10;
	for (int i = 0; i < count; i++){
		int j = 0;
		j = ++i;
		printf("i equals %d %d\n", i,j);

	}
}
