#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>
#include "_maxflow.h"
#include "netio.h"

void check_labels(network * n, char * s, int stop);
void check_tree(network * n, char * s, int stop);
void check_excess(network * n, char * s, int stop);
void check_bounds(network *n, char * s, int stop);

DOUBLE count_excess_vert(network * n, vertex *v);

void check_current(network * n, char * s, int stop);
void check(network * n, char * s, int stop);
int  check_node(vertex * fn, char * s);
void check_vert_node(char *s);
void good_heap(network * n, fib_heap * h, vertex * v, char * s, int stop);
void check_heap(network * n, vertex * w, char * s, int stop);
fib_heap * which_heap(network * n, vertex * v);

#endif
