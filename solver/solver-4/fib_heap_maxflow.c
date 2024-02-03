/*== Fibonacci heap routines for the maxflow.c program ===*/

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
#include "maxflow_typedef.h"
#include "_fib_heap_maxflow.h"
/*
#include "debug.h"
*/
#define check_vert_node(a);
#define check_node(a,b);

/*=================================================================*/
void __cascade(vertex *root, vertex *i)
{
	vertex * fat;
	
	while ((fat = i->hfather)){
		if ( !(i->hmark)){
			i->hmark = MARKED;
			return;
		}
		
		DEL_CHLIST(i, fat);
		ADD_LIST(i, root);
		i->hfather = NULL;
		i->hmark = UN_MARKED;
		i = fat;
	}
} 

/*=================================================================*/
void deckey_fib_heap(fib_heap *fh, vertex *i)
{
  vertex * fat, * fn;
	
  check_vert_node("DEC-1");
	fn = i;

  if ((fat = fn->hfather)){
    
	  DEL_CHLIST(fn, fat);
		ADD_LIST(fn, fh->root);
		fn->hfather = NULL;
		__cascade(fh->root,fat);
	}

	if ( (*fh->cmp)(i, fh->root) < 0 )
	  fh->root = fn;
	check_vert_node("DEC-2");
} 

/*=================================================================*/
void del_fib_heap(fib_heap * fh, vertex * fn)
{
	vertex  * tmp, * fat;
	int i;
	
	check_vert_node("DEL-1");

	if ( fn == fh->root){   
		delmin_fib_heap(fh);
		check_vert_node("DEL-0");
		return;
	}

	if ((fat = fn->hfather) == NULL){        /* fn is in the root-list */
		
		DEL_LIST(fn);
		if (fn->hchild != NULL){
			for (i = 0, tmp = fn->hchild; i < fn->hrank; i++, tmp = tmp->hnext)
			  tmp->hfather = NULL;
			MELD_LIST(fh->root, fn->hchild);
		}
	}else{                             /* fn is not in the root-list */ 

		DEL_CHLIST(fn, fat);

		__cascade(fh->root,fat);
		if (fn->hchild != NULL){
			for (i = 0, tmp = fn->hchild; i < fn->hrank; i++, tmp = tmp->hnext)
			  tmp->hfather = NULL;
			MELD_LIST(fh->root, fn->hchild);
		}
	}

	free(fn);
	check_vert_node("DEL-2");
} 

/*================================================================*/
vertex * delmin_fib_heap(fib_heap * fh)
{
	vertex * linkfib, * root, * tmp;
	int rank, i;
	check_vert_node("DELMIN-1");

	if ( (root = fh->root) == NULL){
		return NULL;
	}


    rank = root->hrank;

	if (root->hnext == root){     /* Just one root */
		if (rank ){
			linkfib = root->hchild;       /* root->hchildlist has father-pointers
											are remaining !! */
			for (i = 0, tmp = linkfib; i < rank; i++, tmp = tmp->hnext)
			  tmp->hfather = NULL;
		}else{                           /* Delete the last node */
			fh->root = NULL;
			free(root);
			check_vert_node("DELMIN-2");
			return root;
		}
	}else{                       /* Not the only root */
		linkfib = root->hnext;
		if ( rank ){               
			for (i = 0, tmp = root->hchild; i < rank; i++, tmp = tmp->hnext)
			  tmp->hfather = NULL;
			REPLACE_LIST(root, root->hchild); /* Pull up the root->hchildlist */
		}else{                           /* no children of the root */
			DEL_LIST(root);              /* remove from the list */
		}

	}
	

	free(root);
	fh = __linking_step(fh, linkfib);
	
	check_vert_node("DELMIN-3");

	return root;
} 

/*==================================================================*/
vertex * findmin_fib_heap(fib_heap *fh)
{
	return fh->root ? fh->root : NULL ;
} 

/*==================================================================*/
void ins_fib_heap(fib_heap *fh, vertex *fn)
{
	check_vert_node("INS-1");

	fn->hprev = fn->hnext = fn;
	fn->hfather = fn->hchild = NULL;
	fn->hrank = 0;
	fn->hmark = UN_MARKED;

	fh->root = __meld_node(fh,fh->root, fn);
	
	check_vert_node("INS-2");
} 

/*==================================================================*/
vertex * __link_fib(fib_heap * fh, vertex * fn1, vertex * fn2)
                                   /* Hangs the bigger root under the smaller */
{
	if ( (*fh->cmp)(fn1, fn2) >= 0 ){   /* fn2 will be the father */
		DEL_LIST(fn1);
		fn2->hrank++;
		fn1->hfather = fn2 ;
		if ( fn2->hchild == NULL ){   /* fn1 will be the only child */
			fn2->hchild = fn1;
			fn1->hnext = fn1->hprev = fn1;
		}else{                        /* fn1 joins to the childlist */
			ADD_LIST(fn1, fn2->hchild);
		}
		
		return fn2;
	}else{                                          /* fn1 will be the father */
		DEL_LIST(fn2);
		fn1->hrank++;
		fn2->hfather = fn1;
		if ( fn1->hchild == NULL ){
			fn1->hchild = fn2;
			fn2->hnext = fn2->hprev = fn2;
		}else{
			ADD_LIST(fn2, fn1->hchild);
		}		
		return fn1;
	}
} 

/*==================================================================*/
#define N	30	/* size of rank array */

fib_heap * __linking_step(fib_heap *fh, vertex *fn)
                                      /* Linking all the same rank root-trees 
										 until every root has different rank.
										 fn is a node on the root-list.*/
										 
{
	static vertex * rank_arr[N];
	vertex  * eor             /* end of roots */
	  ,* actroot                /* actual root  */
		,* lroot                /* linking root */
		  ,* stored, *stored2;
	
	int	i;

	check_vert_node("linking_step");
	
	for(i = 0; i < N; i++)
	  rank_arr[i] = NULL;
	
	eor = fn->hprev;
	stored2 = fn;
	
	do{
		actroot = stored = stored2;
		stored2 = stored->hnext;
		
		while((lroot = rank_arr[actroot->hrank]) != NULL ){
			                         /* actroot and lroot have the same rank */
			rank_arr[actroot->hrank] = NULL;
			actroot = __link_fib(fh, lroot, actroot);
		}
		
		rank_arr[actroot->hrank] = actroot;

		if ((*fh->cmp)(actroot, fn) <= 0 )
		  fn = actroot;

			
	}while (stored != eor);

	check_vert_node("linking_step-1");
	fh->root = fn;
	return fh;
} 

/*==================================================================*/
vertex * __meld_node(fib_heap *fh, vertex *fn1, vertex *fn2)
                                    /* Melds two root_lists, returns the min */
{

	check_vert_node("MELD-1");

	if (fn1 == NULL)
	  return fn2;
	if (fn2 == NULL)
	  return fn1;

	MELD_LIST(fn1, fn2);

	check_vert_node("MELD-2");

	if ( (*fh->cmp)(fn1, fn2) > 0 )
	  return fn2;
	return fn1;
} 

/*================================================================*/
fib_heap * new_heap(cmp_fib_func cmp)
{
	fib_heap * h;
	
	h = (fib_heap *)malloc(sizeof(fib_heap));
	h->cmp = cmp;
	h->root= NULL;
	return h;
} 
