/* This file contains the PLED Maxflow Algorithm developed by 
   Cheriyan & Hagerup. For details look: 
   CH2806-8/89/0000/0118/$01.00(c)1989IEEE */

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

#include "_maxflow.h"
#include "macros.h"

#ifdef DEBUG
#include "debug.h"
#endif

static int Cut, quiet;
static permute_fct Permute;
int It;

/*==================================================================*/
int maxflow(network * n, controll * cont)
                             /* Finds the max-flow in the initialized 
								network n using the PLED alg.  */
{
	vertex *v;
	int pf, sf, rf, cf, g, last;
	int big = 1, stall = 0, was_change = 0, relab = 0;
	DOUBLE old_fl;   
	int j;

	Cut = 0;
	It = 0;
	last = 0;

	switch (cont->perm){
	  case 0:
		Permute = NULL;
		break;
	  case 1:
		Permute = permute_rnd;
		break;
	  default:
		break;
	}
	
	pf = cont->print_freq;
	sf = cont->stall_freq;
	rf = cont->relab_freq;
	cf = cont->cut_freq;
	quiet = cont->quiet;
	Gap = cont->gap;

	while (Delta != 0){
		It++;

		v = select();
		if (v == NULL)
		  break;

		old_fl = Sink->excess;
		g = macro_push(v);


		if (Sink->excess == old_fl)
		  stall++;
		else{
			old_fl = Sink->excess;
			was_change = 1;
			stall = 0;
		}

		if ((was_change && sf && stall >= sf)
			|| (cf && Cut >= cf)
			|| (rf && rf <= It - last)
			|| (g && Gap)){ 
			
			print_report();
			PRINT( quiet, ("BIG_RELABEL...\n"));
			
			stall = 0;
			relab = 0;
			was_change = 0;
			Cut = 0;
			last = It;
			
			if (big_relabel(n, big)){
				PRINT( quiet, ("MINCUT!\n"));
				big = 0;
			}
		}
			
		if (pf && It%pf == 0)
		  print_report();
	}
	
	n->maxflow = Sink->excess;
	restore_flow(n);
	return It;
}

/*==================================================================*/
void heap_insert(vertex * v)
{

	if ( v->excess >= Delta){
		INSERT(v, D_heap);
		
	}else{
		INSERT(v, E_heap);
	}
}

/*==================================================================*/
int macro_push(vertex * v)
                           /* Sends flow from v until v is relabelled 
							  or limflow(v) decreases to < delta/2
							  due to saturating pushes 
							  or until flow is sent from v over a path
							  in the dyn_tree 
							  or until CUT(v) is executed.
							  Returns true if gap is found.*/
{
	edge ** ce;
	int big = 0, vl;

	if (v != FIND_ROOT(v)){
		tree_push(v);
	}else{
		while ((ce = v->current) != NULL &&  
			   RESCAP(v, *ce) <= LIMFLOW(v))
		  {
			  saturate(v, *ce);
		  }
		
		
		if (ce == NULL){

			if (Gap && v->label < Vertnum){ 
				
				(Labarr[(vl = v->label)])--;
				relabel(v);
				
				if (v->label < Vertnum){ 
					(Labarr[v->label])++;
					if (v->label > Labarr[0])
					  Labarr[0] = v->label;
				}else{
					if (vl == Labarr[0] && Labarr[vl] == 0)
					  (Labarr[0])--;
				}
				
				if (Labarr[vl] == 0 && vl < Labarr[0])
				  big = 1;
				
			}else

			relabel(v);
		}else{
/* 22222 */
			 if (LIMFLOW(v) >= Delta/2) {
				LINK(v, OTHER(v, *ce), RESCAP(v, *ce));
				tree_push(v);
			}
		}
	}

	heap_insert(v);		

	if (big)
	  return 1;
	return 0;
}

/*==================================================================*/
vertex * select(void)   
                           /* Selects an active vertex v with 
							  e(v) >= Delta and minimum label,
							  or decrease delta and selects v as
							  before, or stop.
							  Returns NULL if maxflow is reached.*/
{
	vertex * v;

	if (IS_EMPTY(D_heap)){
		if (v = FIND_MIN(E_heap)){ 
		  Delta = MIN2(Delta/2, v->excess);
		}else
		  Delta = 0;

		if (Delta == 0)
		  return NULL;
		
		while ((v = FIND_MIN(E_heap)) && v->excess >= Delta){
			v = DELETE_MIN(E_heap);
			INSERT(v, D_heap);
		}
	}
	v = DELETE_MIN(D_heap);

	return(v);
}

/*==================================================================*/
void tree_push(vertex * v)  
                           /* Cut and saturate the tree edge in the
							  dyn_tree nearest to v whose value is
							  <= LIMFLOW(v), and then send LIMFLOW(v)
							  units of flow from v to FIND_ROOT(v).
							  (v is not in either heap) */
{
	vertex * x;
	edge ** cex;
	DOUBLE lf;
	int i = 0, del = 0;
	
	x = FIND_BOTTLENECK(v, (lf = LIMFLOW(v)));
	
	if (x != v && x != Source && x != Sink && x->excess < Delta)
	  {
		  del = 1;
		  DELETE(x, E_heap);
	  }

	if (x != FIND_ROOT(x)){  /* *(x->current) is the bottleneck */
		tree_cut(x);
		i = 1;
		saturate(x, *(x->current));
	}
	
	if (x == v)
	  return;

	ADD_VALUE(v, -lf);      /* a nonsaturating push from v to x */
 	x->excess += lf;
	v->excess -= lf;

/*	if (i && (cex = x->current)){ 
		LINK(x, OTHER(x, *cex), RESCAP(x, *cex));
	}
*/
	if (del){
		heap_insert(x);
	}
}

/*==================================================================*/
void tree_cut(vertex * v)       /* Cuts the tree at v and counts the 
								   flow on (v ,p(v)) */
{
	edge * e;
	DOUBLE c;

	e = *(v->current);

	c = FIND_VALUE(v);

	if (OUT_EDGE(v, e))
	  e->flow = e->cap - c;
	else
	  e->flow = c;

	CUT(v);
}

/*==================================================================*/
void saturate(vertex * v, edge * e)
                               /* Push RESCAP(e) units of flow from v 
								  over e;
								  (v is either e->from or e->to)
								  (v is not in either heap)
								  (v is a tree-root!)*/
{
	DOUBLE c;
	vertex * w;
	
	c = RESCAP(v, e);

	v->excess -= c;
	
	if (OUT_EDGE(v, e)){
		e->flow += c;
		(w = e->to)->excess += c;
	}else{
		e->flow -= c;
		(w = e->from)->excess += c;
	}

	if (w != Source && w != Sink){
		if (w->excess < Delta){
			DECREASE_KEY(w, E_heap);
		}else if (w->excess - c < Delta){ 
			DELETE(w, E_heap);
			INSERT(w, D_heap);
		}
	}
	next_current(v);
	if (v->current == NULL)
	  Cut++;
}

/*==================================================================*/
void next_current(vertex * v)   /* Takes the next edge adjacent to v,
								   which has RESCAP(v, e) > 0 and
								   d(v) = d(w) + 1 */
{
	edge ** ce, *e;
	edge ** cend = &(v->edgelist[v->degree]);
	int vl = v->label - 1;
	
	for (ce = v->current ? v->current + 1: v->edgelist, v->current = NULL
		 ; ce != cend; ce++){

		e = *ce;
		if (OUT_EDGE(v, e)){ 
			if (e->cap > e->flow && e->to->label == vl){ 
				v->current = ce;
				break;
			}
		}else{ 
			if (e->flow > 0  && e->from->label == vl){ 
				v->current = ce;
				break;
			}
		}
	}
}

/*==================================================================*/
void relabel(vertex * v)
{
	int i, l, j;
	edge * e, ** cew, ** ep;
	vertex * w;

	for (v->label = Vertnum * 3, i = 0, ep = v->edgelist, j = 0
		 ; i < v->degree; i++){

		e = ep[i];
		w = OTHER(v, e);
		if (( cew = w->current) && OTHER(w, *cew) == v){     
                               /* Cutting the adjacent vertex w, where
								  current edge of w points to v */
			if (w != FIND_ROOT(w)){ 
				tree_cut(w);
				j = 1;
			}
			
			next_current(w);
/* 3333333 */
			if ( !(cew = w->current))
			  Cut++;
		}

		if ((l = w->label) < v->label && RESCAP(v, e) > 0)
		  v->label = l;
	}
	v->label += 1;
	
/*  !!!! Permute !!! */

	if (Permute)
	  (*Permute)(v);
	
	next_current(v);

	if (!(cew = v->current))
	  Cut++;
}

/*==================================================================*/
void restore_flow(network * n)    
                       /* Restores the flow values from the tree */
{
	int i;
	vertex * v;
	edge ** ce;

	for(i = 1; i <= Vertnum; i++){
		v = &(n->verts[i]);
		if ((ce = v->current) && FATHER(v) == OTHER(v, *ce)){
			tree_cut(v);
		}
	}
}
/*=================================================================*/
void print_report(void)
{
	PRINT( quiet, ("It: %5d  Flow:%ld  Delta:%10d\n"
		   , It, (int)(Sink->excess), (int)Delta));
}
