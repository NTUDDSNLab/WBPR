/* maxflow_typedef.h == Type definitions for a directed graph +
                        fibonacci heap declarations */

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


#ifndef MAXFLOW_TYPE_H
#define MAXFLOW_TYPE_H

#include <stdio.h>

/*==================================================================*/
typedef struct VERTEX{
	
	/*-- dyn_tree pointers -------------------------*/
	struct VERTEX * dleft, * dright, * dfather;
	DOUBLE dmin;
	DOUBLE dval;
	
	/*-- fibonacci heap pointers -------------------*/
	struct VERTEX	* hnext, * hprev;
	struct VERTEX	* hchild,* hfather;
	int				hrank;
	char			hmark;
	/*----------------------------------------------*/

	struct EDGE ** edgelist;  /* Pointer to the list of pointers to 
								 the adjacent edges. 
								 (No matter that to or from edges) */

	struct EDGE ** current;   /* Pointer to the current edge */

	DOUBLE excess;     /* Excess in the pre-flow */

	int degree;        /* Number of adjacent edges (both direction) */

#ifdef DEBUG
	int index;

	int label;         /* Distance label */
#else
	int label;         /* Distance label */
#endif

}vertex;
/*==================================================================*/
typedef struct EDGE{
	struct VERTEX	* from;
	struct VERTEX	* to;

	DOUBLE flow;       /* Flow value */
	int    cap;        /* Capacity */
}edge;

/*==================================================================*/
typedef  void (* permute_fct)(vertex * v);
/*==================================================================*/
typedef struct NETWORK{

	struct NETWORK	* next, * prev;

	int vertnum;
	int edgenum;

	vertex	* verts; /* Vertex array[1..vertnum] */
	edge    * edges; /* Edge array[1..edgenum] */

	vertex	* source; /* Pointer to the source */
	vertex	* sink;   /* Pointer to the sink */

	DOUBLE maxflow;   /* the value of the maximum flow */

}network;

/*=================================================================*/
typedef struct Controll{
	int perm;
	int stall_freq;
	int relab_freq;
	int cut_freq;
	int print_freq;
	int quiet;
	int gap;
}controll;	
#endif 

