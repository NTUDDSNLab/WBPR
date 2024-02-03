/* dyn_tree.c == Dynamic tree routines, see dyn_tree_maxflow.h */

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

#include "_dyn_tree_maxflow.h"
#include "dyn_tree_maxflow.h"
#include "macros.h"
#include <stdio.h>


/*==================================================================*/
void dyn_link(vertex * or, vertex * nf, DOUBLE new_value)  
	                                /* Hang the tree rooted in oldroot
									   to newfather. Must be in 
									   different trees! */
{
	if (or->dfather)
	  return;

	dyn_splay(nf);

	if (nf->dfather == or){
		printf("Error in dyn_link: Both in the same tree!\n");
		return;

		exit(1);
	}
	or->dfather = nf;	
	or->dval = new_value;
	or->dmin = 0;
}
/*==================================================================*/
void dyn_cut(vertex * c)   /* Cut the tree between cuthere 
							  and its father */

{
	vertex * r, * l;

	if (c->dfather == NULL)
	  return;

	dyn_splay(c);

	if ((r = c->dright)){
		r->dfather = c->dfather;
		c->dright = NULL;
		dvr += dvc;
	}
	if ((l = c->dleft)){
		c->dleft = NULL;
		dvl += dvc;
	}
	c->dfather = NULL;
	dmc = 0;
}

/*==================================================================*/
vertex * dyn_find_root( vertex * r) /* Find the root of item's tree */

{
	return( r->dfather ? dyn_splay(r)->dfather : r);
}

/*==================================================================*/
void dyn_add_value(vertex * f, DOUBLE value)  
                              /* Add value to each node 
								 on the path from 'from' 
								 to the root */

{
	DOUBLE tmp;
	vertex * m, * r;

	dyn_splay(f);

	tmp = (r = f->dright) ? dmr - dvr : 0;
	dmf = MAX2( 0, tmp);
	if ((m = f->dleft)){
		dvf += value;
		dvm -= value;
		dmf = MAX2( dmf, dmm - dvm);
	}else
	  dvf += value;
}

/*==================================================================*/
DOUBLE dyn_find_value(vertex * n)   /* Find the value of item */

{
	if (n->dfather == NULL)
	  return n->dval;
	
	dyn_splay_solid( n);
	return( n->dval);
}

/*==================================================================*/
vertex * dyn_find_bottleneck(vertex * f, DOUBLE neck)
                                /* Find the bottleneck on the path 
								   from 'from' to its root. That is
								   return the nearest ancestor of
								   'from', whose value is <= neck.
								   Otherwise return the root. */
{
	int i;
	vertex * c, * l;  /* f == from
						 c == candidate
						 l == c->dleft */

	DOUBLE valc;

	if ( f->dfather == NULL)                /* from is The Root. */
	  return f;             

	dyn_splay(f);

	if (dvf - dmf > neck)
	  return (f->dfather);                  /* No bottleneck */

	if (dvf <= neck)
	  return (f);                           /* from is the bottleneck */

	if ( (c = f->dright) == NULL || (valc = dvf + dvc) - dmc > neck)
	  return (f->dfather);                  /* No bottleneck */

	/* There is a bottleneck in the right subtree of f 
	   (rooted with c) */

	l = c->dleft;
	
	while (( i = (l ? valc + dvl - dml <= neck : 0)) || valc > neck)
	  { 
		  if (i)
			c = l;
		  else
			c = c->dright;
		  
		  valc += dvc;
		  l = c->dleft;
	  }
	
	return (dyn_splay(c));
}
/*==================================================================*/
vertex * dyn_find_father(vertex * n)
{
	vertex * r;

 	if (n->dfather == NULL)
	  return NULL;
	
	dyn_splay_solid(n);

	if (r = n->dright){
		while (r->dleft)
		  r = r->dleft;
		return r;
	}else
	  return (n->dfather);
}
