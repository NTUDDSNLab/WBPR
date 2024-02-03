/* queue data structure routines */

/* Copyright:
	This program was written by 

	Tamas Badics, 1991,
	Rutgers University, RUTCOR
	P.O.Box 5062
	New Brunswick, NJ, 08903
	e-mail: badics@rutcor.rutgers.edu
 
	The code may be used and modified for not-for-profit use.
	This notice must be remained.	
====================================================================*/

#include "queue.h"
#include <stdio.h>

queue * new_queue(int size)
{
	queue * q;
	
	q = (queue *)malloc(sizeof(queue));
	q->first = q->last = q->start 
	  = (queue_node *)calloc(size, sizeof(queue_node));
	q->end = &q->start[size];

	return q;
}

void init_queue(queue * q)
{
	q->first = q->last = q->start;
}

void free_queue(queue * q)
{
	free(q->start);
	free(q);
}

void enqueue(queue * q, void * item)
{
	q->last->data = item;

	if (q->last == q->end)
	  q->last = q->start;
	else
	  q->last++;
}

void * dequeue(queue * q)
{
	void * data;
	
	data = q->first->data;
	
	if (q->first == q->end)
	  q->first = q->start;
	else
	  q->first++;
	
	return(data);
}
