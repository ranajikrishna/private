/**
 * Given a string s, find the length of the longest substring without repeating 
 * characters.
 * */


#include <stdio.h>
#include <stdlib.h>


int lengthOfLongestSubstring(char * s){
	char *address[128] = {NULL};
	int len =0;
	char *tmp,*start=s; //Assign the first pointer
	while(*start)
	{
		tmp = address[*start]; //Extract value
		address[*start] = start; //Assign the value
		if(tmp >= s)
		{
			len = len > start - s ? len : start - s; //Difference between current pointer and starting pointer
			s = tmp +1; // next starting pointer
		}
		start++;
	}
	len = len > start - s ? len : start -s;
	return len;
}

int main(int argc, char* argv[])
{
	//	char *p = "aaa";
	char *p = "aaabcdefghijkdlmnop";
	//	char *p = "aaabcdefghijddlmnop";
	//	char *p = "aaabcdefghijkalmnop";
	//	char *p = "aaabcdefghijkblmnop";
	//	char *p = "aaabcdefgdijkblmnop";
	//	char *p = "aaabcdefghijkblmnopqrstuvwxyz";
	//char *p = "pwwkew";
	//char *p = "bbbbbbbb";
	int prnt = lengthOfLongestSubstring(p);
	printf("%d \n", prnt);

	return (0);
}
