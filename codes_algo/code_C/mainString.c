
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]){

//	char tmpString [99] = "Ranaji Krishna";
//	printf("%s:",tmpString);

	char tmpString [99];
	printf("Enter a string:");
	//scanf("%s", tmpString);
	gets(tmpString);

	for (int i =0; i < 99; i++){
		if(*(tmpString+i) == '\000'){
			printf("%d\n", i);
			break;
		}
	}
}
