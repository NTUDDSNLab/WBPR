/*=================================================================*/
/*== Initialization ===============================================*/
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
#include "netio.h"
#include "stopper1.h"

#define FEAS 1
#define NOFEAS 0

/*=== Global variable definitions =========*/

vertex * Source, * Sink;
int Vertnum, Edgenum, Itnum;
char Gap;

fib_heap * E_heap = NULL;
fib_heap * D_heap = NULL;
DOUBLE Delta;

queue * Q;
int * Labarr;    /* array of existing labels [0..Vertnum]:
					Labarr[i] = # of vertices whose label is i */


#ifdef DEBUG
network * net;
#endif

/*=========================================*/

int maxflow_warm_start(network * n, int need_preflow, controll * cont);
void init_preflow(network * n);

/*=================================================================*/
int maxflow_cold_start(char * prob_file, network ** n, controll * cont)
                           /* Reads the problem from the file.
							  Then start the PLED algorithm.
							  Return false if infeasible.
							  Return the network via n.
							  See maxflow_typedef.h for controll.*/
{
	int quiet = cont->quiet;
	int ret;
	
	PRINT( quiet, ("I am reading the network...\n"));
	*n = read_net(prob_file);

#ifdef DEBUG
	net = *n;
#endif
	
	start_stopper();
	
	ret = maxflow_warm_start(*n, 1, cont);
	printf( "Runtime: %.2f\n", elapsed_stopper_time());
	
	return ret;
}	

/*=================================================================*/
int maxflow_warm_start(network * n, int need_preflow, controll * cont)
	                          /* After the allocations, with
								 a prepared network * n, and
								 a given preflow, it will
								 start the maxflow algorithm */
{
	vertex * nv, * v; 
	int i; 
	int quiet = cont->quiet;
	
	PRINT( quiet, ("Initialize is started\n"));
	
	/*---- Fill up the global variables ----*/
	n->maxflow = 0;
	Source = n->source;
	Sink = n->sink;
	Vertnum = n->vertnum;
	Edgenum = n->edgenum;
	Itnum = 0;

	/*---- Default values of the frequencies ----*/
	if (cont->stall_freq == -1)
	  if (cont->gap == 0)
		cont->stall_freq = Vertnum / 2;
	  else
		cont->stall_freq = 0;
	
	if (cont->relab_freq == -1)
	  cont->relab_freq = 1.5*Vertnum;
	if (cont->cut_freq == -1)
	  cont->cut_freq = 0;
	if (cont->gap == -1)
	  cont->gap = 1;
	
	Gap = cont->gap;

	if (need_preflow)
	  init_preflow(n);
	else{ 
		Delta = 0; 
		/*---- Counting the maximal excess for Delta ----------*/
		for( i = 1, nv = n->verts; i <= Vertnum; i++){
			v = &(nv[i]);
			if (v->excess > Delta && v != Sink && v!= Source)
			  Delta = v->excess;
		}
	}

	PRINT( quiet, ("Initialize is finished\n"));
	
	if (Delta == 0 && Sink->excess == 0)     
	  return NOFEAS;              /* No feasible solution */ 

	if (Gap)
	  Labarr = (int *)calloc(Vertnum, sizeof(int));

	Q = new_queue(2*Vertnum);

	if (big_relabel(n, 1) && Sink->excess == 0)
	  return NOFEAS;              /* No feasible solution */ 

	Itnum = maxflow(n, cont);

	return FEAS;
}
/*==================================================================*/
void init_preflow(network * n)
	                 /*Saturate all the outgoing edges from source
					   while counting the maximal excess for Delta*/
{
	int i;
	edge * e;
	
	Delta = 0;
	
	for( i = 0; i < Source->degree; i++){
		e = Source->edgelist[i];
		if( OUT_EDGE(Source, e)){
			e->flow = e->cap;
			e->to->excess += e->flow;
			if (e->to->excess > Delta && e->to != Sink)
			  Delta = e->to->excess;
		}
	}
}

