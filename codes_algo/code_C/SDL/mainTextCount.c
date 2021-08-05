
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CHARS 200

typedef char string[MAX_CHARS+1];  

int main(){
   	int i = 0;		// word counter.
   	string array[100];	// store words. 

   	FILE *my;		
   	my = fopen("textFile.txt","r");		// Parse file.
   	while(1){
    		fscanf(my, "%s", array[i]);	// Read file & store words.
		i++;
		if (feof(my)){
			break;
		}
   	}
   	fclose(my);

	int freqArray [200] = {0}; 		 // Initialize array to store frequency of words.
	int l = 0;				 // Indicator (= 1 : word already counted).
	for (int j =0; j < (i-1); j++){		 // Iterarte through words.

		for (int k = 0; k < (i-1); k++){ // Check for word match.

			if ( (strcmp(array[j],array[k]) == 0)){
				if (j<k){	       // Word already counted.
					l = 1;
					break;
				}
				if (j==k && l == 1){   // Word already counted.
					break;
				}
				*(freqArray + j) += 1; // New word.	
			}
		}
		if (l == 0){	
			printf("%s : %d \n", array[j], *(freqArray + j)); // Print if new word.
		}
		l = 0;
	}
}	
