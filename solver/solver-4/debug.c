/* debug.c -- checking routines for PLED */
#ifdef DEBUG

#include <stdio.h>
#include "debug.h"
#include "_maxflow.h"

extern network * net;
extern vertex * Source, * Sink;

void check_labels(network * n, char * s, int stop)
{
	int i, j;
	vertex * v, *w;
	edge * e;
	DOUBLE rc;
	
	for (i = 1; i <= n->vertnum; i++){
		v = &(n->verts[i]);
		for (j = 0; j < v->degree; j++){
			e = v->edgelist[j];
			w = OTHER(v, e);
			if (v->label > w->label + 1) {
				if (w != FATHER(v)){
					rc = RESCAP(v, e);
				}else{ 
					rc = FIND_VALUE(v);
				}
				if (rc > 0){ 
					fprintf(stderr, "Err: LABEL in ");
					fprintf(stderr, "%s\n", s);
					print_vertex(v, 0);
					print_vertex(w, 0);
					if (stop)
					  exit(-1);
				}
			}
		}
	}
}


void check_tree(network * n, char * s, int stop)
{
	int i;
	vertex * v, *w;

	for (i = 1; i <= n->vertnum; i++){
		v = &(n->verts[i]);
		if (v != FIND_ROOT(v)){
			w = FATHER(v);
			if (OTHER(v,*(v->current)) != w){
				fprintf(stderr, "Err: TREE in ");
				fprintf(stderr, "%s\n", s);
				print_vertex( v, 0);
				print_vertex( w, 0);
				if (stop)
				  exit(-1);
			}
		}
	}
}
void check_current(network * n, char * s, int stop)
{
	int i;
	vertex * v, *w;

	for (i = 1; i <= n->vertnum; i++){
		v = &(n->verts[i]);
		if (v->current){
			w = OTHER(v, *(v->current));
			if (w->label != v->label - 1){
				fprintf(stderr, "Err: CURRENT in ");
				fprintf(stderr, "%s\n",s);
				print_vertex( v, 0);
				print_vertex( w, 0);
				if (stop)
				  exit(-1);
			}
		}
	}
}


int check_node(vertex * fn, char * s)
{
	vertex * eor, * ff;
	int i;
	
	
	if (!fn || !fn->hnext || !fn->hprev)
	  return 0;
	
	for(eor = fn->hprev, ff = fn, i = 1; ff != eor; ff = ff->hnext, i++){
		if (ff != ff->hnext->hprev){
			fprintf(stderr, "Err: fib_list-1 in ");
			print_err(s, 1);
		}
		
	}
	
	for(eor = fn->hnext, ff = fn; ff != eor; ff = ff->hprev){
		if (ff != ff->hprev->hnext){
			fprintf(stderr, "Err: fib_list-2 in ");
			print_err(s, 1);
		}
	}
	return i;
}

void check_roots(vertex * fn, char * s)
{
	vertex * eor, * ff;
	int i;
	
	if (fn == NULL)
	  return;
		
	for(eor = fn->hprev, ff = fn, i = 1; ff != eor; ff = ff->hnext, i++){
		if (ff->hfather){
			fprintf(stderr, "Err: Rootlist in ");
			print_err(s, 1);
		}
	}
}

void check_vert_node(char * s)
{
	vertex * fn;
	vertex * v;
	int i, j;

	check_roots(E_heap->root, s);
	check_roots(D_heap->root, s);
	
	for (i = 1; i <= net->vertnum; i++){
		v = &(net->verts[i]);

		fn = v;

	    j = check_node(fn, s);
		if (j && fn->hfather && fn->hfather->hrank != j){
			fprintf(stderr, "Err: Wrong rank in ");
			print_err(s, 1);
		}
		if (fn->hchild && fn->hchild->hfather != fn){
			fprintf(stderr, "Err: Wrong parent in ");
			print_err(s, 1);
		}
		
	}
}
DOUBLE count_flow(vertex * w, edge * e, DOUBLE rc)
{
	DOUBLE f;
	
	if (OUT_EDGE(w, e))
	  f = e->cap - rc;
	else
	  f = rc;

	return f;
}


void check_bounds(network *n, char * s, int stop)
{
	int i;
	DOUBLE flow, rc;
	edge * e;
	vertex * v, * w;
	
	for (i = 1; i <= n->edgenum; i++){
		e = &(n->edges[i]);
		v = e->from;
		w = e->to;
		flow = e->flow;
		
		if (FATHER(v) == w || FATHER(w) == v ){
			if (v->current && *(v->current) == e){
				rc = FIND_VALUE(v);
				flow = count_flow(v, e, rc);
			}else if (w->current && *(w->current) == e){
				rc = FIND_VALUE(w);
				flow = count_flow(w, e, rc);
			}
		} 

		if ((flow < 0) || (e->cap < flow)){
 
			fprintf(stderr, "Err: wrong bounds! in ");

			print_err(s, stop);
		}
	} 
}

DOUBLE count_excess_vert(network *n, vertex *v)
{
	DOUBLE rc,f,ex = 0;
	edge * e;
	vertex * w, *vf, *wf;
	int j;
	
	for (j = 0; j < v->degree; j++){
		e = v->edgelist[j];
		w = OTHER(v, e);
		vf = FATHER(v);
		wf = FATHER(w);
		
		if (w == vf && *(v->current) == e){
			rc = FIND_VALUE(v);
			f = count_flow(v, e, rc);
		}else if (v == wf && *(w->current) == e){ 
			rc = FIND_VALUE(w);
			f = count_flow(w, e, rc);
		}else{ 
			rc = RESCAP(v, e);
			f = count_flow(v, e, rc);
		}
		if (OUT_EDGE(v, e))
		  ex -= f;
		else
		  ex += f;
	}		

	return ex;
}

void check_excess(network * n, char * s, int stop)
{
	int i;
	vertex * v;
	DOUBLE ex;
	
	for (i = 1; i <= n->vertnum; i++){
		v = &(n->verts[i]);
		if (v != Source){ 
			ex = count_excess_vert(n, v);
			if (ex != v->excess){ 
				fprintf(stderr, "Err: Wrong excess at");
				print_vertex( v, 0);
				fprintf(stderr, "Real: %d Stored: %d\n", (int)ex, (int)v->excess);
				
				print_err(s, stop);
			}
		}
	}
}		

void check(network * n, char * s, int stop)
{

	fprintf(stderr, "Checking...\n");
	
	check_labels(n, s, stop);
	check_tree(n, s, stop);
	check_current(n, s, stop);
	check_excess(n, s, stop);
	check_bounds(n, s, stop);
	
	if (D_heap->root && D_heap->root->hnext == NULL){ 
		fprintf(stderr, "Err: droot");
		print_err(s, 1);
	}
	
	if (E_heap->root && E_heap->root->hnext == NULL){ 
		fprintf(stderr, "Err: eroot");
		print_err(s, 1);
	}
}

fib_heap * which_heap(network * n, vertex * v1)
{
	vertex * eor, * ff, *v = v1;

	
	if (v->hprev == NULL)
	  return NULL;
	
	for (; v->hfather; v = v->hfather);

	eor = v->hprev;
	ff = eor;
	do{	
		ff = ff->hnext;
		if (ff == D_heap->root)
		  return D_heap;
		else if (ff == E_heap->root)
		  return E_heap;
	}while (ff != eor);
	return NULL;
}

void good_heap(network * n, fib_heap * h, vertex * v, char * s, int stop)
{
	fib_heap * h1;
	
	h1 = which_heap(n, v);
	
	if (h != h1){
		fprintf(stderr, "Err: In wrong heap\n");
		print_err(s, stop);
	}
}

void check_heap(network * n, vertex * w, char * s, int stop)
{
	int i, j = 0;
	vertex * v;
	fib_heap * h;
	
	
	for (i = 1; i <= n->vertnum; i++){
		v = &(n->verts[i]);
		if (v != Source && v!= Sink){ 
			h = which_heap(n, v);
			if (h == NULL)
			  if ( v != w){ 
				  fprintf(stderr, "%d Not in heap at %s:\n",j++, s);
				  print_vertex( v, 0);
				  print_err("", 1);
			  }else
				continue;
			
			if (v->excess >= Delta){ 
				if (h != D_heap){ 
					fprintf(stderr, "Err: In wrong heap\n");
					print_vertex( v, 0);
					print_err(s, stop);
				}
			}else if (h != E_heap){ 
				fprintf(stderr, "Err: In wrong heap\n");
				print_vertex( v, 0);
				print_err(s, stop);
			}
		}
	}
}

#endif
