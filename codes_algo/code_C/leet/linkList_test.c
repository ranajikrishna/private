/* https://www.youtube.com/watch?v=VOpjAHCee7c
 * */

#include <stdio.h>
#include <stdlib.h>
#include "myAlgoPrep.h"	


struct node{
	int value;
	struct node* next;
}; 

typedef struct node node_t;

node_t *create_new_node(int value){
/**
 * We use *malloc* inside a fxn., instead of just using static memo. allocation, 
 * since static allocation is local to the fxn. This means that 
 * the same memo. address could be assigned every time the allocation 
 * is made. This would result in overwritting the existing 
 * memo. with new data and the same address being returned by the 
 * fxn. Dynamic allocation with *malloc* uses the *heap* to store the 
 * values, resulting in the allocation not being local to the fxn.
 * */
	node_t *new_node = malloc(sizeof(node_t));	
	new_node->value = value;
	new_node->next = NULL;
	return new_node;
}

void printlist(node_t *head){
	node_t *tmp = head;
	
	while (tmp != NULL){
	printf("%d - ", tmp->value);
	tmp = tmp->next; 
	}
	printf("\n");
}

int main(int argc, char* argv[])
{
	node_t n1, n2, n3, n4;
	node_t *head;

	n1.value = 45;
	n2.value = 8;
	n3.value = 32;
	n4.value = 78;

	//link them up
	head = &n3; 
	n3.next = &n4;
	n4.next = &n2;
	n2.next = &n1;
	n1.next = NULL; //so we know when to stop

	node_t *start = NULL;
	node_t *tmp;
	tmp = create_new_node(10);
	tmp->next = start;
	start = tmp;
	tmp = create_new_node(15);
	tmp->next = start;
	start = tmp;


	// To create and populate a list in a loop.
	node_t *start = NULL;
	node_t *tmp;
	for (int i=0; i<25; i++) {
		tmp = create_new_node(i);
		tmp->next = start;
		start = tmp;
	}

/**
 * Note: This will NOT work, because the NULL pointer, `n1`, does 
 * not point to an address that contains a node of type `node_t`. 
 * To use a pointer to assign values to a node, the node has to be 
 * declared, a ptr. should point to it and then be used to assign 
 * values to it. To just use pointers one has to use the `malloc` 
 * fxn. 
*/
	// This will NOT work
	node_t *n1 = NULL;
	n1->value = 45
	n1->next = NULL;

	// This WILL work.
	node_t n1;
	node_t *p1 = &n1;
	p1->value = 45
	p1->next = NULL;

	// This WILL work as well.
	node_t *p1 = malloc(sizeof(node_t));
	p1->value = 45
	p1->next = NULL;

	printlist(start);


	return (0);
}
