/* dyn_tree_maxflow.h == Dynamic tree data structure type definitions 

   Based on Sleator, Tarjan : Self-Adjusting Binary Search Trees
   JACM Vol. 32., No. 3, July 1985 pp.652-686 
   
   Slight difference: The Real roots have only middle children,
   so that infinite value on the root can be handled */

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


#ifndef DYN_T_
#define DYN_T_

#include "maxflow_typedef.h"

void dyn_make_tree(vertex * item, DOUBLE value);
                                        /* Put item to a new 1-node
										   tree with value */

void dyn_link(vertex * oldroot, vertex * newfather, DOUBLE new_value);
                                    /* Hang the tree rooted in oldroot
									   to newfather with new_value. 
									   Must be in different trees! */

void dyn_cut(vertex * cuthere);       /* Cut the tree between cuthere 
										 and its father */

void dyn_add_value(vertex * from, DOUBLE value);    
                                        /* Add value to each node 
										   on the path from 'from' 
										   to the root */

DOUBLE dyn_find_value(vertex * item); /* Find the value of item */

vertex * dyn_find_bottleneck(vertex * from, DOUBLE neck);
                                 /* Find the bottleneck on the path 
									from 'from' to its root. That is
									return the nearest ancestor of
									'from', whose value is <= neck. */

vertex * dyn_find_root(vertex * item);
                                  /* Find the root of item's tree */

vertex * dyn_find_father(vertex * item);
                                  /* Find the father in the tree */

#endif
