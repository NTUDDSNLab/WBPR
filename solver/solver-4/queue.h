/* Queue data structure for breadth first search */

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

#ifndef _QUEUE
#define _QUEUE

typedef struct QUEUE_NODE
{
	void * data;
}queue_node;

typedef struct QUEUE
{
	struct QUEUE_NODE * first;
	struct QUEUE_NODE * last;
	struct QUEUE_NODE * start;
	struct QUEUE_NODE * end;
}queue;

queue * new_queue(int size);
void init_queue(queue * q);
void free_queue(queue * q);
void enqueue(queue * q, void * item);
void * dequeue(queue * q);
#define IS_EMPTY_QUEUE(q) q->first == q->last

#endif
