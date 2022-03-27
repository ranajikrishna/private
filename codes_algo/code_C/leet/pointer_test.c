
#include <stdio.h>
#include <stdlib.h>


struct Index {
	char *str;
};

typedef struct Index ind;

int fxn_one(char *s)
{
	ind *sub;
	sub->str = s;

	return (0);
}


int main(int argc, char* argv[])
{
	char *p = "test";
	int ret = fxn_one(p);

	return (0);
}
