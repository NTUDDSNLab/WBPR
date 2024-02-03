/* Data types and global declarations 
   for the Cheriyan-Hagerup Max-flow algorithm */

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

#ifndef _MAXFLOW_H
#define _MAXFLOW_H


/*==================================================================*/
/*========  NETWORK type for the algorithm  ========================*/
/*========  see maxflow_typedef.h           ========================*/
/*==================================================================*/

#include <stdio.h>
#include "maxflow_typedef.h"
#include "dyn_tree_maxflow.h"
#include "queue.h"

/*==================================================================*/
typedef int (*cmp_fib_func)(vertex * p1, vertex * p2);
/*==================================================================*/

typedef struct FIB_HEAP{
	vertex * root;
	cmp_fib_func cmp;
}fib_heap;
/*==================================================================*/
/*====  Heap routines   ============================================*/

void     ins_fib_heap(fib_heap *fh, vertex * v);
vertex * findmin_fib_heap(fib_heap *fh);
vertex * delmin_fib_heap();
void     deckey_fib_heap(fib_heap *fh, vertex * v);
void     del_fib_heap(fib_heap *fh, vertex * v);
fib_heap * new_heap(cmp_fib_func cmp);

/*==================================================================*/
/*======  Global variables          ================================*/
/*==================================================================*/

extern DOUBLE Delta;
extern fib_heap * E_heap;
extern fib_heap * D_heap;
extern int Vertnum, Edgenum;
extern int It;
extern vertex * Source, * Sink;
extern int * Labarr;
extern queue * Q;
extern char Gap;

/*==================================================================*/
/*======  Macro definitions         ================================*/
/*==================================================================*/


#define PRINT(quiet, args) 	if (!(quiet)) printf args

#define OUT_EDGE(V, E) ((E)->from == V)

#define RESCAP(V, E) ((OUT_EDGE(V, E)) ? ((E)->cap - (E)->flow) : (E)->flow)

#define OTHER(V, E)  (OUT_EDGE(V, E) ? (E)->to : (E)->from)

/* 111111 */

#define LIMFLOW(V) (((V)->excess >= 2*Delta) ? (DOUBLE)(((V)->excess/2)) : (V)->excess)
/*
#define LIMFLOW(V) (V)->excess
*/
/*==================================================================*/
/*======  Dynamic_tree operations   ================================*/
/*==================================================================*/

/* 
vertex * FIND_ROOT(vertex *);
DOUBLE   FIND_VALUE(vertex *); 
void     ADD_VALUE(vertex *, DOUBLE );
void     LINK(vertex *, vertex *,  DOUBLE);
void     CUT(vertex *);
vertex * FIND_BOTTLENECK(vertex *, DOUBLE)
vertex * FATHER(vertex *);
*/

#define FIND_ROOT(V)     dyn_find_root(V)

#define FIND_VALUE(V)    dyn_find_value(V) 

#define ADD_VALUE(V, C)  dyn_add_value(V, C) 

#define LINK(V, W, C)    dyn_link(V, W, C) 

#define CUT(V)           dyn_cut(V) 

#define FIND_BOTTLENECK(V, C)  dyn_find_bottleneck(V, C)

#define FATHER(V)        dyn_find_father(V)

#define EMPTY_DNODE(V) { (V)->dleft = (V)->dright = (V)->dfather=NULL;\
					     (V)->dval = (V)->dmin = 0; } 

/*==================================================================*/
/*======  Heap operations   ========================================*/
/*==================================================================*/

/*
void     INSERT(vertex *, heap *);
void     DELETE(vertex *, heap *);
vertex * FIND_MIN(heap *);
vertex * DELETE_MIN(heap *);
void     DECREASE_KEY(vertex *, heap *) ;
int      IS_EMPTY(heap *);
*/

#define INSERT(V, H)  ins_fib_heap(H, V)

#define DELETE(V, H)  del_fib_heap(H, V)

#define FIND_MIN(H)   findmin_fib_heap(H) 

#define DELETE_MIN(H) delmin_fib_heap(H)

/*!!!!!!!!!! before calling 'DECREASE_KEY' !!!!!!!!!!!!!!!!!*/
/*!!!!!!!! the key of V must be decreased !!!!!!!!*/

#define DECREASE_KEY(V, H) deckey_fib_heap(H, V) 

#define IS_EMPTY(H)   (H->root == NULL)

#define EMPTY_HNODE(V) { (V)->hnext = (V)->hprev = (V)->hfather \
                       = (V)->hchild = NULL;\
					     (V)->hmark = (V)->hrank = 0; } 

/*==================================================================*/
/*========   Function declarations   ===============================*/
/*==================================================================*/

int maxflow(network * n, controll * cont);
                              /* Find the max-flow in the initialized 
								 network n using the PLED alg. 
								 Return the # of iterations*/

int cmp_excess(vertex * v1, vertex * v2);
int cmp_label(vertex * v1, vertex * v2);

void init_heaps(network * n);  /* Initialize the e_heap and d_heap
								  with the given preflow */

void init_tree(network * n, int maxex);   /* After big_relabel, link all the
								  possible vertices */

void heap_insert(vertex * v);
                              /* Insert v into E_heap 
								 if v->excess < Delta , 
								 otherwise insert into D_heap*/

int macro_push(vertex * v); 
                            /* Send flow from v until v is relabelled 
							   or limflow(v) decrease to < delta/2
							   due to saturating pushes 
							   or until flow is sent from v over 
							   a path in the dyn_tree 
							   or until CUT(v) is executed. */

vertex * select(void);      /* Select an active vertex v with 
							   e(v) >= Delta and minimum label,
							   or decrease delta and select v as
							   before, or stop.
							   Return NULL if maxflow is reached.*/

void tree_push(vertex * v);
                            /* Cut and saturate the tree edge in the
							   dyn_tree nearest to v whose value is
							   <= LIMFLOW(v), and then send LIMFLOW(v)
							   units of flow from v to FIND_ROOT(v).
							   (v is not in either heap) */

void tree_cut(vertex * v);        /* Cut the tree at v and counts the 
									 flow on (v ,p(v)) */

void saturate(vertex * v, edge * e);
                            /* Push RESCAP(e) units of flow from v 
							   over e;
							   (v is either e->from or e->to)
							   (v is not in either heap) */

void next_current(vertex * v);    /* Take the next edge adjacent to v,
									 which has RESCAP(v, e) > 0 */

void relabel(vertex * v);
                            /* Relabel v and do permutation on
							   v->edgelist */

void restore_flow(network * n);   
                           /* Restore the flow values from the tree */

void permute_rnd(vertex * v);     
                           /* Permute the edgelist of v randomly */
 
int big_relabel(network *n, int how_big);  
                           /* Bfs from the sink, searching for the
							  min_cut if exists already. Returns
							  true if mincut is found.
							  Finally search from the source on 
							  the left partition of the cut 
							  (not necessarely min).
							  Relabels all the vertices.
							  Does an init_heap, and init_tree.
							  Calling freqency is heuristic.*/

int search(vertex * source, queue * q, int start_label);
                        /* Bfs from source using q, and start_label */

void print_report(void);

#endif
