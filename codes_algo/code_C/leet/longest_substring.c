/**
 * Given a string s, find the length of the longest substring without repeating 
 * characters.
 * */


#include <stdio.h>
#include <stdlib.h>

struct Index {
	char *str;
	char *stp;
	char *set;
	int len;
};

typedef struct Index ind;

ind *chk_rep(ind *stg) 
{
	char *j = stg->str + 1;
	while (stg->stp - j >= 0) {
		if (*(stg->str) != *j) {
			j += 1;
		} else {

			// Create left string.
			ind *lft = malloc(sizeof(stg));
			if (j - stg->str != 1) {
				lft->str = stg->str;
				lft->stp = j-1;
				lft->set = stg->set;
			} else { // Back-to-back repeat.
				lft->str = j;
				lft->stp = stg->stp;
				lft->set = j;
			}

			// Create right string.
			ind *rgt = malloc(sizeof(stg));	
			rgt->str = stg->str;
			rgt->stp = stg->stp;
			lft->set = stg->str;

			// Execute longer str. first.
			if (lft->stp - lft->str > (rgt->stp - rgt->str && stg->len)){ 
				stg = chk_rep(lft);
				stg = chk_rep(rgt);

			} else {
				stg = chk_rep(rgt);
				stg = chk_rep(lft);
			}
			return (stg);
		} 
	} 
	
	if (stg->stp - stg->str > 0) {
		stg->str += 1;
		stg = chk_rep(stg);
	}

	// Update length.
	if (stg->str - stg->set > stg->len){
		stg->len = stg->str - stg->set; 
	}
	return (stg);
}

int lengthOfLongestSubstring(char *s)
{
	
	int l = 0;
	int rep = 1;
	do {
		if (*s != *(s+l) && rep==1) {
			rep = 0;
		}
		l++;
	} while (*(s+l) != '\0');

	ind *string;
	string->str = s;
	string->stp = s+l-1;
	string->set = s;
	string->len = 0; 
	
	string = chk_rep(string);

	return (0);
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
