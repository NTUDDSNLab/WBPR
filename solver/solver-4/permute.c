/* Edgelist permutation for a vertex */

#include "_maxflow.h"
#include "math_to_gcc.h"

void permute_rnd(vertex * v)  /* Permute the edgelist of v randomly */
{
	int i, j, k, deg;
	edge * tmp;
	
	deg = v->degree;
	
	for (i = 0; i < deg; i++){
		k = deg-i;
		j = random(k);
		tmp = v->edgelist[j];
		v->edgelist[j] = v->edgelist[deg - i - 1];
		v->edgelist[deg - i - 1] = tmp;
	}
	v->current = NULL;
}

