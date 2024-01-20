/*** Copyright 1989 Norbert Schlenker.  All rights reserved.

 *** This software is distributed subject to the following provisions:
 ***    - this notice may not be removed;
 ***    - you may modify the source code, as long as redistributed
 ***      versions have their modifications clearly marked;
 ***    - no charge, other than a nominal copying fee, may be made
 ***      when providing copies of this source code to others;
 ***    - if this source code is used as part of a product which is
 ***      distributed only as a binary, a copy of this source code
 ***      must be included in the distribution.
 ***
 *** Unlike the GNU GPL, use of this code does not obligate you to
 *** disclose your own proprietary source code.

 *** The author of this software provides no warranty, express or
 *** implied, as to its utility or correctness.  That said, reports
 *** of bugs or compatibility problems will be gladly received by
 *** nfs@princeton.edu, and fixes will be attempted.
 ***/


/*** netgen - C version of the standard NETGEN network generator 
 ***          This program is a functional equivalent of the
 ***          standard network generator NETGEN described in:
 ***		Klingman, D., A. Napier, and J. Stutz, "NETGEN:  A Program
 ***		  for Generating Large Scale Capacitated Assignment,
 ***		  Transportation, and Minimum Cost Flow Network Problems",
 ***		  Management Science 20, 5, 814-821 (1974)
 ***
 ***	      This software provides a number of interfaces for use by
 ***	      network solvers.  Standard call interfaces are supplied for
 ***	      use by (Unix) C and Fortran solvers, with generation parameters
 ***	      passed into the generator and the flow network passed back to
 ***	      the solver via large external (COMMON in Fortran) arrays.
 ***	      For the DIMACS challenge, this code will produce output files
 ***	      in the appropriate format for later reading by solvers.
 ***          Undefine the symbol DIMACS when using the call interface.
 ***
 ***          The generator produces exact duplicates of the networks
 ***          made by the Fortran code (even though that means bugs
 ***          are being perpetuated). It is faster by orders of magnitude
 ***          in generating large networks, primarily by imposing the
 ***          notion of the abstract data type INDEX_LIST and implementing
 ***          the type operations in a reasonably efficient fashion.
 ***/

/*** Generates transportation problems if:
 ***	SOURCES+SINKS == NODES && TSOURCES == TSINKS == 0
 ***
 *** Generates assignment problems if above conditions are satisfied, and:
 ***	SOURCES == SINKS && SUPPLY == SOURCES
 ***
 *** Generates maximum flow problems if not an assignment problem and:
 ***	MINCOST == MAXCOST == 1

 *** Implementation notes:
 ***
 ***	This file contains both a Fortran and a C interface. The
 ***	Fortran interface is suffixed with an underscore to make it
 ***	callable in the normal fashion from Fortran (a Unix convention).
 ***
 ***    Because Fortran has no facility for pointers, the common arrays
 ***    are statically allocated.  Static allocation has nothing to recommend
 ***    it except for the need for a Fortran interface.
 ***
 ***    This software expects input parameters to be long integers
 ***    (in the sense of C); that means no INTEGER*2 from Fortran callers.
 ***
 ***	Compiling with -DDIMACS produces a program that reads problem
 ***	parameters, generates the appropriate problem, and prints it.
 ***
 ***	Compiling with -DDEBUG produces code with externally visible
 ***	procedure names, useful for debugging and profiling.
 ***/


/*** System interfaces */

#include <stdio.h>


/*** Public interfaces */

#define ALLOCATE_NETWORK
#include "netgen.h"

#define PROBLEM_PARMS 13		/* aliases for generation parameters */
#define NODES	    parms[0]		/* number of nodes */
#define SOURCES     parms[1]		/* number of sources (including transshipment) */
#define SINKS	    parms[2]		/* number of sinks (including transshipment) */
#define DENSITY     parms[3]		/* number of (requested) arcs */
#define MINCOST     parms[4]		/* minimum cost of arcs */
#define MAXCOST     parms[5]		/* maximum cost of arcs */
#define SUPPLY	    parms[6]		/* total supply */
#define TSOURCES    parms[7]		/* transshipment sources */
#define TSINKS	    parms[8]		/* transshipment sinks */
#define HICOST	    parms[9]		/* percent of skeleton arcs given maximum cost */
#define CAPACITATED parms[10]		/* percent of arcs to be capacitated */
#define MINCAP	    parms[11]		/* minimum capacity for capacitated arcs */
#define MAXCAP	    parms[12]		/* maximum capacity for capacitated arcs */


/*** Private interfaces */

#ifdef DEBUG
#define PRIVATE
#else
#define PRIVATE static
#endif

#ifdef __STDC__
PRIVATE void create_supply(NODE, CAPACITY); /* create supply nodes */
PRIVATE void create_assignment(long*);	/* create assignment problem */
PRIVATE void sort_skeleton(int);	/* sorts skeleton chains */
PRIVATE void pick_head(long*, int, NODE); /* choose destination nodes for rubbish arcs */
PRIVATE void error_exit(long);		/* print error message and exit */
#else
PRIVATE void create_supply();		/* create supply nodes */
PRIVATE void create_assignment();	/* create assignment problem */
PRIVATE void sort_skeleton();		/* sorts skeleton chains */
PRIVATE void pick_head();		/* chooses destination nodes for rubbish arcs */
PRIVATE void error_exit();		/* print error message and exit */
#endif

/*** Private variables */

static NODE nodes_left;
static ARC arc_count;
static NODE pred[MAXARCS];
static NODE head[MAXARCS];
static NODE tail[MAXARCS];


/*** Local macros */

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define SAVE_ARC(tail, head, cost, capacity)	/* records an arc where our caller can get it */ \
  {				\
    FROM[arc_count] = tail;	\
    TO  [arc_count] = head;	\
    C   [arc_count] = cost;	\
    U   [arc_count] = capacity; \
    arc_count++;		\
  }


/*** Fortran callable interface routine */

void netgen_(seed, parms, generated_nodes, generated_arcs)
long* seed;			/* pointer to random seed */
long parms[PROBLEM_PARMS];	/* problem parameters */
long* generated_nodes;		/* number of generated nodes */
long* generated_arcs;		/* number of generated arcs */
{
  *generated_nodes = NODES;
  if ((*generated_arcs = netgen(*seed, parms)) < 0)
    error_exit(*generated_arcs);
}


/*** C callable interface routine */

ARC netgen(seed, parms)
long seed;			/* random seed */
long parms[];			/* problem parameters */
{
  register NODE i,j,k;
  NODE source;
  NODE node;
  NODE sinks_per_source;
  NODE* sinks;
  NODE it;
  int chain_length;
  COST cost;
  CAPACITY cap;
  INDEX_LIST handle;
  int supply_per_sink;
  int partial_supply;
  int sort_count;


/*** Perform sanity checks on the input */

  if (seed <= 0)
    return BAD_SEED;
  if (NODES > MAXNODES || DENSITY > MAXARCS)
    return TOO_BIG;
  if ((NODES <= 0) ||
      (NODES > DENSITY) ||
      (SOURCES <= 0) ||
      (SINKS <= 0) ||
      (SOURCES + SINKS > NODES) ||
      (MINCOST > MAXCOST) ||
      (SUPPLY < SOURCES) ||
      (TSOURCES > SOURCES) ||
      (TSINKS > SINKS) ||
      (HICOST < 0 || HICOST > 100) ||
      (CAPACITATED < 0 || CAPACITATED > 100) ||
      (MINCAP > MAXCAP))
    return BAD_PARMS;


/*** Do a little bit of setting up. */

  set_random(seed);

  arc_count = 0;
  nodes_left = NODES - SINKS + TSINKS;

  if ((SOURCES-TSOURCES)+(SINKS-TSINKS) == NODES &&
      (SOURCES-TSOURCES) == (SINKS-TSINKS) &&
       SOURCES == SUPPLY) {
    create_assignment(parms);
    return arc_count;
  }

  (void)memset((void *)B, 0, sizeof(B));/* set supplies and demands to zero */

  create_supply((NODE)SOURCES, (CAPACITY)SUPPLY);


/*** Form most of the network skeleton.  First, 60% of the transshipment
 *** nodes are divided evenly among the various sources; the remainder
 *** are chained onto the end of the chains belonging to random sources.
 ***/

  for (i = 1; i <= SOURCES; i++)	/* point SOURCES at themselves */
    pred[i] = i;
  handle = make_index_list((INDEX)(SOURCES + 1), (INDEX)(NODES - SINKS));
  source = 1;
  for (i = NODES-SOURCES-SINKS; i > (4*(NODES-SOURCES-SINKS)+9)/10; i--) {
    node = choose_index(handle, (INDEX)random(1L, (long)index_size(handle)));
    pred[node] = pred[source];
    pred[source] = node;
    if (++source > SOURCES)
      source = 1;
  }
  for ( ; i > 0; --i) {
    node = choose_index(handle, (INDEX)random(1L, (long)index_size(handle)));
    source = random(1L, SOURCES);
    pred[node] = pred[source];
    pred[source] = node;
  }
  free_index_list(handle);


/*** For each source chain, hook it to an "appropriate" number of sinks,
 *** place capacities and costs on the skeleton edges, and then call
 *** pick_head to add a bunch of rubbish edges at each node on the chain.
 ***/

  for (source = 1; source <= SOURCES; source++) {
    sort_count = 0;
    node = pred[source];
    while (node != source) {
      sort_count++;
      head[sort_count] = node;
      node = tail[sort_count] = pred[node];
    }
    if ((NODES-SOURCES-SINKS) == 0)
      sinks_per_source = SINKS/SOURCES + 1;
    else
/* changing to handle overflows with large n; Mar 18 -- jc */
      sinks_per_source = ((double) 2*sort_count*SINKS) / ((double) NODES-SOURCES-SINKS);
    sinks_per_source = MAX(2, MIN(sinks_per_source, SINKS));
    sinks = (NODE*) malloc(sinks_per_source * sizeof(NODE));
    handle = make_index_list((INDEX)(NODES - SINKS), (INDEX)(NODES - 1));
    for (i = 0; i < sinks_per_source; i++) {
      sinks[i] = choose_index(handle, (INDEX)random(1L, (long)index_size(handle)));
    }
    if (source == SOURCES && index_size(handle) > 0) {
      sinks = (NODE*) realloc((void *)sinks, (sinks_per_source + index_size(handle)) * sizeof(NODE));
      while (index_size(handle) > 0) {
	j = choose_index(handle, 1);
	if (B[j] == 0)
	  sinks[sinks_per_source++] = j;
      }
    }
    free_index_list(handle);

    chain_length = sort_count;
    supply_per_sink = B[source-1] / sinks_per_source;
    k = pred[source];
    for (i = 0; i < sinks_per_source; i++) {
      sort_count++;
      partial_supply = random(1L, (long)supply_per_sink);
      j = random(0L, (long)sinks_per_source - 1);
      tail[sort_count] = k;
      head[sort_count] = sinks[i] + 1;
      B[sinks[i]] -= partial_supply;
      B[sinks[j]] -= (supply_per_sink - partial_supply);
      k = source;
      for (j = random(1L, (long)chain_length); j > 0; j--)
	k = pred[k];
    }
    B[sinks[0]] -= (B[source-1] % sinks_per_source);
    free((void *)sinks);

    sort_skeleton(sort_count);
    tail[sort_count+1] = 0;
    for (i = 1; i <= sort_count; ) {
      handle = make_index_list((INDEX)(SOURCES - TSOURCES + 1), (INDEX)NODES);
      remove_index(handle, (INDEX)tail[i]);
      it = tail[i];
      while (it == tail[i]) {
	remove_index(handle, (INDEX)head[i]);
	cap = SUPPLY;
	if (random(1L, 100L) <= CAPACITATED)
	  cap = MAX(B[source-1], MINCAP);
	cost = MAXCOST;
	if (random(1L, 100L) > HICOST)
	  cost = random(MINCOST, MAXCOST);
	SAVE_ARC(it,head[i],cost,cap);
	i++;
      }
      pick_head(parms, handle, it);
      free_index_list(handle);
    }
  }


/*** Add more rubbish edges out of the transshipment sinks. */

  for (i = NODES - SINKS + 1; i <= NODES - SINKS + TSINKS; i++) {
    handle = make_index_list((INDEX)(SOURCES - TSOURCES + 1), (INDEX)NODES);
    remove_index(handle, (INDEX)i);
    pick_head(parms, handle, i);
    free_index_list(handle);
  }

  return arc_count;
}


PRIVATE void create_supply(sources, supply)
NODE sources;
CAPACITY supply;
{
  CAPACITY supply_per_source = supply / sources;
  CAPACITY partial_supply;
  NODE i;

  for (i = 0; i < sources; i++) {
    B[i] += (partial_supply = random(1L, (long)supply_per_source));
    B[random(0L, (long)(sources - 1))] += supply_per_source - partial_supply;
  }
  B[random(0L, (long)(sources - 1))] += supply % sources;
}


PRIVATE void create_assignment(parms)
long parms[];
{
  INDEX_LIST skeleton, handle;
  INDEX index;
  NODE source;

  for (source = 0; source < NODES/2; source++)
    B[source] = 1;
  for ( ; source < NODES; source++)
    B[source] = -1;

  skeleton = make_index_list((INDEX)(SOURCES + 1), (INDEX)NODES);
  for (source = 1; source <= NODES/2; source++) {
    index = choose_index(skeleton, (INDEX)random(1L, (long)index_size(skeleton)));
    SAVE_ARC(source, index, random(MINCOST, MAXCOST), 1);
    handle = make_index_list((INDEX)(SOURCES + 1), (INDEX)NODES);
    remove_index(handle, index);
    pick_head(parms, handle, source);
    free_index_list(handle);
  }
  free_index_list(skeleton);
}


PRIVATE void sort_skeleton(sort_count) 		/* Shell sort */
int sort_count;
{
  int m,i,j,k;
  int temp;

  m = sort_count;
  while ((m /= 2) != 0) {
    k = sort_count - m;
    for (j = 1; j <= k; j++) {
      i = j;
      while (i >= 1 && tail[i] > tail[i+m]) {
	temp = tail[i];
	tail[i] = tail[i+m];
	tail[i+m] = temp;
	temp = head[i];
	head[i] = head[i+m];
	head[i+m] = temp;
	i -= m;
      }
    }
  }
}


PRIVATE void pick_head(parms, handle, desired_tail)
long parms[];
INDEX_LIST handle;
NODE desired_tail;
{
  NODE non_sources = NODES - SOURCES + TSOURCES;

/* changing Aug 29 -- jc
  ARC remaining_arcs = DENSITY - arc_count;
*/
  int  remaining_arcs = (int) DENSITY - (int) arc_count;

  INDEX index;
  int limit;
  long upper_bound;
  CAPACITY cap;

/* changing Aug 29 -- jc
*/
  nodes_left--;
  if ((2 * (int) nodes_left) >= (int) remaining_arcs)
    return;

  if ((remaining_arcs + non_sources - pseudo_size(handle) - 1) / (nodes_left + 1) >= non_sources - 1) {
    limit = non_sources;
  } else {
    upper_bound = 2 * (remaining_arcs / (nodes_left + 1) - 1);
    do {
      limit = random(1L, upper_bound);
      if (nodes_left == 0)
	limit = remaining_arcs;
/* changing to handle overflows with large n; Mar 18 -- jc */
    } while ( ((double) nodes_left * (non_sources - 1)) < ((double) remaining_arcs - limit));
  }

  for ( ; limit > 0; limit--) {
    index = choose_index(handle, (INDEX)random(1L, (long)pseudo_size(handle)));
    cap = SUPPLY;
    if (random(1L, 100L) <= CAPACITATED)
      cap = random(MINCAP, MAXCAP);

/* adding Aug 29 -- jc */
if ((1 <= index) && (index <= NODES)) {
    SAVE_ARC(desired_tail, index, random(MINCOST, MAXCOST), cap);
    }

  }
}


/*** Print an appropriate error message and then exit with a nonzero code. */

PRIVATE void error_exit(rc)
long rc;
{
  switch (rc) {
    case BAD_SEED:
      fprintf(stderr, "NETGEN requires a positive random seed\n");
      break;
    case TOO_BIG:
      fprintf(stderr, "Problem too large for generator\n");
      break;
    case BAD_PARMS:
      fprintf(stderr, "Inconsistent parameter settings - check the input\n");
      break;
    case ALLOCATION_FAILURE:
      fprintf(stderr, "Memory allocation failure\n");
      break;
    default:
      fprintf(stderr, "Internal error\n");
      break;
  }
  exit(1000 - (int)rc);
}

#ifdef DIMACS			/* generates network on standard output */

#define READ(v) 		     	/* read one variable using scanf */	\
	switch( scanf("%ld", &v) ) {						\
		case 1:								\
			break;							\
		default:							\
			exit(0);						\
		}

void main()
{
  long seed;
  long problem;
  long parms[PROBLEM_PARMS];
  long arcs;
  int i;

/*** Read problem parameters and generate networks */

  while (1) {
    READ(seed);
    if (seed <= 0) exit(0);
    READ(problem);
    if (problem <= 0) exit(0);
    for (i = 0; i < PROBLEM_PARMS; i++)
      READ(parms[i]);
    printf("c NETGEN flow network generator (C version)\n");
    printf("c  Problem %2ld input parameters\n", problem);
    printf("c  ---------------------------\n");
    printf("c   Random seed:          %10ld\n",   seed);
    printf("c   Number of nodes:      %10ld\n",   NODES);
    printf("c   Source nodes:         %10ld\n",   SOURCES);
    printf("c   Sink nodes:           %10ld\n",   SINKS);
    printf("c   Number of arcs:       %10ld\n",   DENSITY);
    printf("c   Minimum arc cost:     %10ld\n",   MINCOST);
    printf("c   Maximum arc cost:     %10ld\n",   MAXCOST);
    printf("c   Total supply:         %10ld\n",   SUPPLY);
    printf("c   Transshipment -\n");
    printf("c     Sources:            %10ld\n",   TSOURCES);
    printf("c     Sinks:              %10ld\n",   TSINKS);
    printf("c   Skeleton arcs -\n");
    printf("c     With max cost:      %10ld%%\n", HICOST);
    printf("c     Capacitated:        %10ld%%\n", CAPACITATED);
    printf("c   Minimum arc capacity: %10ld\n",   MINCAP);
    printf("c   Maximum arc capacity: %10ld\n",   MAXCAP);

    if ((arcs = netgen(seed, parms)) < 0)
      error_exit(arcs);
    if ((SOURCES-TSOURCES)+(SINKS-TSINKS) == NODES &&
	(SOURCES-TSOURCES) == (SINKS-TSINKS) &&
	 SOURCES == SUPPLY) {
      printf("c\n");
      printf("c  *** Assignment ***\n");
      printf("c\n");
      printf("p asn %ld %ld\n", NODES, arcs);
      for (i = 0; i < NODES; i++) {
	if (B[i] > 0)
	  printf("n %ld\n", i + 1);
      }
      for (i = 0; i < arcs; i++) {
	printf("a %ld %ld %ld\n", FROM[i], TO[i], C[i]);
      }
    } else
    if (MINCOST == 1 && MAXCOST == 1) {
      printf("c\n");
      printf("c  *** Maximum flow ***\n");
      printf("c\n");
      printf("p max %ld %ld\n", NODES, arcs);
      for (i = 0; i < NODES; i++) {
	if (B[i] > 0)
	  printf("n %ld s\n", i + 1);
	else
	if (B[i] < 0)
	  printf("n %ld t\n", i + 1);
      }
      for (i = 0; i < arcs; i++) {
	printf("a %ld %ld %ld\n", FROM[i], TO[i], U[i]);
      }
    } else {
      printf("c\n");
      printf("c  *** Minimum cost flow ***\n");
      printf("c\n");
      printf("p min %ld %ld\n", NODES, arcs);
      for (i = 0; i < NODES; i++) {
	if (B[i] != 0)
	  printf("n %ld %ld\n", i + 1, B[i]);
      }
      for (i = 0; i < arcs; i++) {
	printf("a %ld %ld %ld %ld %ld\n", FROM[i], TO[i], 0, U[i], C[i]);
      }
    }
  }
}

#endif
