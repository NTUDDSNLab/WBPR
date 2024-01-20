/*** netgen.h
 *** Prototype code for inclusion into network generation routines
 ***/

/*** Constant definitions */

#ifndef NULL
#define NULL 0
#endif

#define BAD_SEED  -1			/* error indicators */
#define TOO_BIG   -2
#define BAD_PARMS -3
#define ALLOCATION_FAILURE -4


/*** Type definitions */

typedef unsigned long NODE;		/* node number */
typedef unsigned long ARC;		/* arc number */
typedef long CAPACITY;			/* arc capacity */
typedef long COST;			/* arc cost */
typedef unsigned long INDEX;		/* index element */
typedef int INDEX_LIST;			/* index list handle */


/*** Function prototypes */

#ifdef __STDC__

#include <stdlib.h>
#include <string.h>

void netgen_(long*, long[], long*, long*);	/* Fortran external interface */
ARC netgen(long, long*);	    		/* C external interface */

INDEX_LIST make_index_list(INDEX, INDEX); /* allocates a new index list */
void free_index_list(INDEX_LIST);	/* frees an existing list */
INDEX choose_index(INDEX_LIST, INDEX);	/* chooses index at specified position */
void remove_index(INDEX_LIST, INDEX);	/* removes specified index from list */
INDEX index_size(INDEX_LIST); 		/* number of indices remaining */
INDEX pseudo_size(INDEX_LIST);		/* "modified" index size */

void set_random(long);	    		/* initialize random seed */
long random(long, long);    		/* generate random integer in interval */

#else

void *malloc();				/* some standard header should define this */
void *realloc();			/* ditto */
void free();				/* ditto */
void *memset();				/* ditto */
void exit();				/* ditto */

void netgen_();				/* Fortran external interface */
ARC netgen();		    		/* C external interface */

INDEX_LIST make_index_list();		/* allocates a new index list */
void free_index_list();			/* frees an existing list */
INDEX choose_index();			/* chooses index at specified position */
void remove_index();			/* removes specified index from list */
INDEX index_size(); 			/* number of indices remaining */
INDEX pseudo_size();			/* "modified" index size */

void set_random();	    		/* initialize random seed */
long random();    			/* generate random integer in interval */

#endif

/*** To maintain compatibility with the old Fortran network generator,
 *** the following are defined.  This allows linking the generator code
 *** with the solver, with the generated network passed to the solver
 *** through arrays in memory.
 ***/

#define MAXNODES 10000	    		/* maximum problem sizes */
#define MAXARCS  40000

#define FROM arrays_	    		/* aliases for network storage */
#define TO   arraye_
#define U    arrayu_
#define C    arrayc_
#define B    arrayb_

#ifdef ALLOCATE_NETWORK	    		/* storage definitions */
#define EXTERN 
#else
#define EXTERN extern
#endif

EXTERN NODE     FROM[MAXARCS];	/* origin of each arc */
EXTERN NODE     TO  [MAXARCS];	/* destination */
EXTERN CAPACITY U   [MAXARCS];	/* capacity */
EXTERN COST     C   [MAXARCS];	/* cost */
EXTERN CAPACITY B   [MAXNODES];	/* supply (demand) at each node */
