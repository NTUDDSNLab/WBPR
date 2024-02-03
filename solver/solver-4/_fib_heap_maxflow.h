/* _fib_heap_maxflow.h	== _fib_heap data struct. and routines */

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

#ifndef _FIB_H
#define _FIB_H

#include "maxflow_typedef.h"
#include "_maxflow.h"

#define MARKED      1
#define UN_MARKED	0

vertex * __meld_node(fib_heap *fh, vertex *fn1, vertex *fn2);

vertex * __link_fib();
fib_heap * __linking_step(fib_heap *fh, vertex *fn);

void __cascade(vertex *root, vertex *i);

/*====== list routines in macro ======================================*/

#define MELD_LIST(FN1,FN2){	\
		((FN1)->hnext)->hprev = (FN2)->hprev;    \
		((FN2)->hprev)->hnext = (FN1)->hnext;	\
		(FN1)->hnext = (FN2);	\
		(FN2)->hprev = (FN1);	\
}

/* ADD FN to LIST */
#define ADD_LIST(FN, LIST){	\
		(FN)->hprev = (LIST);	\
		(FN)->hnext = (LIST)->hnext;	\
		(FN)->hnext->hprev = (FN);  \
		(LIST)->hnext = (FN);	\
}

/* deletes FN from its list */
#define DEL_LIST(FN){	\
		(FN)->hnext->hprev = (FN)->hprev;	\
		(FN)->hprev->hnext = (FN)->hnext;	\
}

/* Remove FN from its father's childlist */
#define DEL_CHLIST(FN, FAT)  { \
								 if (--((FAT)->hrank)) { \
								    DEL_LIST(FN);\
									(FAT)->hchild = (FN)->hprev; \
							     }else \
									(FAT)->hchild = NULL;\
							 }
							   
													  
/* Replace FN with LIST's list (FN is not lonely!!!) */
#define REPLACE_LIST(FN,LIST){ 	\
		(LIST)->hprev->hnext = (FN)->hnext;	\
		(FN)->hnext->hprev = (LIST)->hprev;	\
		(LIST)->hprev = (FN)->hprev;	\
		(FN)->hprev->hnext = (LIST);	\
}

#define free(F) {\
				   (F)->hfather \
				 = (F)->hchild \
		         = (F)->hnext \
		   		 = (F)->hprev = NULL; \
				   (F)->hmark = (F)->hrank = 0; \
			 }

#endif 


