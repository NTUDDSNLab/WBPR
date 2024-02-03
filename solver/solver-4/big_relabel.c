/*=================================================================*/
/*== Big relabel    ===============================================*/
/*=================================================================*/

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

#include <stdio.h>
#include "_maxflow.h"
#include "dyn_tree_maxflow.h"
#include "queue.h"
#include "netio.h"

/*=================================================================*/
int cmp_excess(vertex * v1, vertex * v2)
{
	if (v2->excess > v1->excess)
	  return 1;
	else if (v2->excess == v1->excess)
	  return 0;
	return -1;
}

int cmp_label(vertex * v1, vertex * v2)
{
	if (v2->label > v1->label)
	  return -1;
	else if (v2->label == v1->label)
	  return 0;
	return 1;
}

/*===============================================================*/
void init_heaps(network * n)     /* Initialize the e_heap and d_heap
									with the given preflow */
{
	vertex * v;
	int i, inf;

	inf = 3 * Vertnum;
	
	for( i = 1; i <= Vertnum; i++){
		v = &(n->verts[i]);
		v->current = NULL;
		next_current(v);
		if ( v->label != inf && v != Source && v != Sink){
			heap_insert(v);
		}
	}
}

/*==============================================================*/
void init_tree(network * n, int maxex)     /*  Build the dyn_tree structure
									with the given labels */
{
	vertex * v, * w;
	edge ** ce;
	int i;

	for( i = 1; i <= Vertnum; i++){
		v = &(n->verts[i]);
		
		if ((ce = v->current) && v != Source && v != Sink
			&& v->label == (w = OTHER(v, *ce))->label + 1
			&& LIMFLOW(v) >= maxex/2 ){
		    LINK(v, w, RESCAP(v, *ce));
		}
	}
}

/*===============================================================*/
static int Maxlev;

int big_relabel(network *n, int big)       
                        /* Bfs from the sink, searching for the
						   min_cut if exists already. Returns
						   true if mincut is found.
						   Finally search from the source on the left
						   partition of the cut (not necessarely min).
						   Relabels all the vertices.
						   Does an init_heap, and init_tree.
						   Calling freqency is heuristic.*/
{
	int i, inf;
	vertex  * v;
	int min_cut;
	DOUBLE c;
	edge * e;
	int maxex = 0;
	

	inf = 3 * Vertnum;

	D_heap->root = E_heap->root = NULL;
	
	if (Gap)
	  Labarr[0] = Maxlev = 0;

	for (i = 1; i <= Vertnum; i++){
		v = &(n->verts[i]);

		if (FATHER(v)){ 
			c = FIND_VALUE(v);
			e = *(v->current);
			if (OUT_EDGE(v, e))
			  e->flow = e->cap - c;
			else
			  e->flow = c;
		}
		if (Gap)
		  Labarr[i] = 0;

		if (v->excess > maxex)
		  maxex = v->excess;
	}

	for (i = 1; i <= Vertnum; i++){
		v = &(n->verts[i]);
		v->label = inf;
		EMPTY_HNODE(v);
		EMPTY_DNODE(v);
	}

	if (big)
	  min_cut = search(Sink, Q, 0);
	else
	  min_cut = 0;
	
	if (Gap)
	  Labarr[0] = Maxlev;

	if (Source->label < Vertnum)
	  Source->label = Vertnum;
	else	
	  search(Source, Q, Vertnum);
		
	init_heaps(n);

	init_tree(n, maxex);

	return (min_cut);
}

/*==================================================================*/
int search(vertex * source, queue * q, int start_label)
{
	int i, level, inf, mincut;
	vertex * v, * w;
	edge * e;

    level = start_label;
	mincut = 1;
	inf = 3 * Vertnum;
	
	init_queue(q);
	enqueue(q, source);
    enqueue(q, NULL);

    while (1){
		v = (vertex *)dequeue(q);
		if (v == NULL){
			if (IS_EMPTY_QUEUE(q))
			  break;
			level++;
			enqueue(q, NULL);
	    }else{
			v->label = level;

			if (Gap && level < Vertnum)
			  (Labarr[level])++;
			
			for (i = 0; i < v->degree; i++){
				e = v->edgelist[i];
				if ((w = OTHER(v, e))->label == inf 
					&& RESCAP(w, e) > 0)
				  {	
					  if (w->excess > 0){
						  mincut = 0;
					  }					
					  w->label = -1;
					  enqueue(q, w);
				  }
			}
	    }
	}

	Maxlev = level;
	
	return (mincut);
}
/*=================================================================*/
int check_maxflow(network * n)
{
	int i, j, inf, cutcap, level;
	vertex  * v, * w;
	edge * e;

	inf = 3 * Vertnum;

	for (i = 1; i <= Vertnum; i++)
	  n->verts[i].label = inf;

	/* A breadth first search from the source through the 
	   residual graph */

    level = 1;

	init_queue(Q);
	enqueue(Q, Source);
    enqueue(Q, NULL);

    while (1){
		v = (vertex *)dequeue(Q);
		if (v == NULL){
			if (IS_EMPTY_QUEUE(Q))
			  break;
			level++;
			enqueue(Q, NULL);
	    }else{
			v->label = level;
			
			for (i = 0; i < v->degree; i++){
				e = v->edgelist[i];
				if ((w = OTHER(v, e))->label == inf 
					&& RESCAP(v, e) > 0)
				  {	
					  w->label = -1;
					  enqueue(Q, w);
				  }
			}
	    }
	} /* end of search */
	
	for (i = 1, cutcap = 0; i <= Vertnum; i ++){  
		                                 /* counting the minimum
											cut value */
		v = &(n->verts[i]);
		if (v->label < inf){
			for (j = 0; j < v->degree; j++){
				e = v->edgelist[j];
				if (e->flow > e->cap)
				  print_err("Err: Capacity bound is hurt!", 0);
				if (OUT_EDGE(v, e) && OTHER(v, e)->label == inf)
				  cutcap += e->cap;
			}
		} 
	}

	if (n->maxflow == cutcap){ 
		return 1;
	}else
	  return 0;
}

