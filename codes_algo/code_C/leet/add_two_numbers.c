/**
 * You are given two non-empty linked lists representing two non-negative integers. 
 * The digits are stored in reverse order, and each of their nodes contains a 
 * single digit. Add the two numbers and return the sum as a linked list.
 *
 * You may assume the two numbers do not contain any leading zero, except the 
 * number 0 itself.
 */


#include <stdio.h>
#include <stdlib.h>
#include "myAlgoPrep.h"	

struct ListNode {
	int val;
	struct ListNode *next;
};

typedef struct ListNode node;

node *add_two_number(node *head_one, node *head_two)
{
	node *sum = malloc(sizeof(node));
	node *head_sum = sum;
	node *del_node;
	int one = head_one->val;
	int two = head_two->val;
	int bit = 0;
	while (head_one!=NULL || head_two!=NULL) {
		sum->val = one + two + bit;
		bit = 0;
		if (sum->val >= 10) {
			sum->val -= 10;
			bit = 1;
		}
		del_node = sum;
		if (head_one==NULL){
			one = 0;
			head_one = NULL;	
		} else {
			head_one = head_one->next;
			if (head_one!=NULL) {
				one = head_one->val;
			} else {
				one = 0;
			}
		}
		if (head_two==NULL){
			two = 0;
			head_two = NULL;	
		} else {
			head_two = head_two->next;
			if (head_two!=NULL) {
				two = head_two->val;
			} else {
				two = 0;
			}
		}
		sum->next = malloc(sizeof(node));
		sum = sum->next;
	}
	if (bit!=0) {
		sum->val = bit;
		sum->next = NULL;
	} else {
		del_node->next = NULL;
	}

	return head_sum; 
};


void printlist(node *head){
	node *tmp = head;
	
	while (tmp != NULL){
	printf("%d - ", tmp->val);
	tmp = tmp->next; 
	}
	printf("\n");
}

int main(int argc, char* argv[])
{   

	// List for first number
	node *first_one  = malloc(sizeof(node));	
	node *second_one = malloc(sizeof(node));	
	node *third_one  = malloc(sizeof(node));	 

	// List for second number
	node *first_two  = malloc(sizeof(node));	
	node *second_two = malloc(sizeof(node));	
	node *third_two  = malloc(sizeof(node));	
	node *fourth_two  = malloc(sizeof(node));	
	node *fifth_two = malloc(sizeof(node));	
	node *sixth_two  = malloc(sizeof(node));	 

	// Populate first list
	node *head_one = first_one;	// assign pointer to locate head
	first_one->val  = 2;
	second_one->val = 4;
	third_one->val  = 5;

	// Populate second list
	node *head_two = first_two; 	// assign pointer to locate head
	first_two->val  = 7;
	second_two->val = 7;
	third_two->val  = 4;
	fourth_two->val = 9;
	fifth_two->val = 9;
	sixth_two->val = 9;

	// First list assign pointers
	first_one->next = second_one;
	second_one->next = third_one;
	third_one->next = NULL;

	// Second list assign pointers
	first_two->next = second_two;
	second_two->next = third_two;
	third_two->next = fourth_two;
	fourth_two->next = fifth_two;
	fifth_two->next = sixth_two;
	sixth_two->next = NULL;

	node *sum = add_two_number(head_one, head_two);

	printlist(sum);

	return(0);
}




