/* Input/output functions for DIMACS standard max-flow format */

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
#include "_maxflow.h"
#include "netio.h"
#include "_dyn_tree_maxflow.h"

#define LENGTH 255

#ifndef DEBUG
#define index label
#endif

/*==================================================================*/
network * read_net(char * filename)  
                              /* Reads a network from DIMACS .max file.
								 Allocates all the necessary arrays*/
{
	FILE * f;
	network * n;
	char line[LENGTH];
	char dum[10];
	int i, j;
	edge * ed;
	edge * ne, * e;
	vertex * nv, *v;
	int c = 0;

	if (filename == NULL){
		f = stdin;
	}else if ((f = fopen(filename, "r")) == NULL) {
		fprintf(stderr,"Read_net: file %s can't be opened\n",filename);
		exit(0);
	}	
	
	while(NULL != fgets( line, LENGTH, f)){

		switch ( line[0] ){
		  case 'c':
			break;
			
		  case 'p':
			n = (network *)malloc(sizeof(network));
			sscanf(line, "%s%s%d%d", dum, dum
				   , &(n->vertnum), &(n->edgenum)); 
			n->verts =(vertex *)calloc(n->vertnum + 1, sizeof(vertex));
			n->edges = (edge *)calloc(n->edgenum + 1, sizeof(edge));
			ed = n->edges + 1;
			break;
			
		  case 'n':
			sscanf(line, "%s%d%s", dum, &i, dum); 
			if (dum[0] == 's'){
				n->source = &((n->verts)[i]);
				break;
			}
			if (dum[0] == 't'){
				n->sink =  &((n->verts)[i]);
				break;
			}
			break;
			
		  case 'a':
			if (c == n->edgenum)
			  print_err("Err: More edges in the input file than given!", 1);
			
			sscanf(line, "%s%d%d%d", dum, &i, &j, &(ed->cap));
			if (i == 0 || j == 0)
			  print_err("Err: 0 index in the input file!", 1);

			if (i > n->vertnum || j > n->vertnum)
			  print_err("Err: Big index in the input file!", 1);
			
			ed->from = &((n->verts)[i]);
			ed->to = &((n->verts)[j]);
			ed->flow = 0;
			((n->verts)[i]).degree++;
			((n->verts)[j]).degree++;
			ed++;
			c++;
			break;
		  default:
			break;
		} /* switch */
	}/* while */
	
	if (f != stdin)
	  fclose(f);
	
	if( c < n->edgenum) 
	  print_err( "Err: Less edges in the input file than given!", 1);

	/*------ Now build the adjacency lists for each vertex ----*/
	for( i = 1, nv = n->verts; i <= n->vertnum; i++){   
                                         /* Edgelist allocation */
		v = &(nv[i]);
		EMPTY_HNODE(v);
		EMPTY_DNODE(v);
		v->current = v->edgelist 
		  = (edge **)calloc(v->degree, sizeof(edge *));

#ifdef DEBUG
		v->index = i;
#endif
	}

	for( i = 1, ne = n->edges; i <= n->edgenum; i++){
                                 /* Fill of the edgelists */ 
		e = &(ne[i]);
		*(e->from->current) = e;
		*(e->to->current) = e; 
		(e->from->current)++;
		(e->to->current)++;
		e->flow = 0;
	}

	/*---- Allocate the two heaps -----*/
	E_heap = new_heap(cmp_excess); 
	D_heap = new_heap(cmp_label);
	
	return n;
}
/*===============================================================*/
void free_net(network * n)
{
	int i;
	
	free(n->edges);
	free(E_heap);
	free(D_heap);
	if (Gap)
	  free(Labarr);
	free_queue(Q);

	for (i = 1; i <= n->vertnum; i++){
		free(n->verts[i].edgelist);
	}
	free(n->verts);
	free(n);
}

/*==================================================================*/
void print_net(FILE * output, network * n, char * s, int quiet
			   , char * prob_name, controll * cont)  
					/* s is a combinations of [s, d, t, a, v, e]*/
{
	int i, k, vnum, e_num;

#ifdef DEBUG
	vertex v;
	int j, to, from;
#endif
	
	vnum = n->vertnum;
	e_num = n->edgenum;

#ifndef DEBUG
	for (i = 1; i <= vnum; i++){
		n->verts[i].index = i;
	} 
#endif
	
	for (k = 0; k < strlen(s); k++){
		switch (s[k]) {

		  case 't':			/* t = tiltle */
			PRINT( quiet, ("\n---------------------------------\n"));
			PRINT( quiet, ("\nVertex num:%6d Edge num: %10d\n", vnum, e_num));
			PRINT( quiet, ("\nSource: %6d Sink: %6d\n"
				   , n->source->index, n->sink->index));
			break;

 		  case 'p':			/* p = parameter settings */
			PRINT( quiet, ("\n---------------------------------\n"));
			PRINT( quiet, ("\nParameters of the algorithm:\n"));
			PRINT( quiet, ("Vertex num:%6d Edge num: %10d\n", vnum, e_num));
			PRINT( quiet, ("Stall_freq: %8d\n", cont->stall_freq));
			PRINT( quiet, ("Relab_freq: %8d\n", cont->relab_freq));
			PRINT( quiet, ("  Cut_freq: %8d\n", cont->cut_freq));
			PRINT( quiet, ("       Gap: %8d\n", cont->gap));
			break;

		  case 's':			/* DIMACS standard output */
			print_DIMACS_std(output, n, prob_name);
			break;

#ifdef DEBUG
		  case 'a':			/* a = adjacency list */
			PRINT( quiet, ("\n---------------------------------\n"));
			PRINT( quiet, ("\n\nThe adjacency lists:\n"));
			for( i = 1; i<= vnum; i++){
				v = n->verts[i];
				PRINT( quiet, ("\nVertex: [%6d]\n", i));
				for( j = 0; j < v.degree; j++){
					from = v.edgelist[j]->from->index;
					to = v.edgelist[j]->to->index;
					PRINT( quiet, ("\t(%6d %6d)\n", from, to));
				}
			}
			break;
			
		  case 'e':			/* e = edges */
			PRINT( quiet, ("\n---------------------------------\n"));
			PRINT( quiet, ("\nThe edges:\n\n"));
			for( i = 1 ; i <= e_num ; i++ ){
				print_edge(&(n->edges[i]));
			}
			break;
			
		  case 'v':			/* v = vertices */
			PRINT( quiet, ("\n---------------------------------\n"));
			PRINT( quiet, ("\nThe vertices:\n\n"));
			for( i = 1; i<= vnum; i++){
				print_vertex(&n->verts[i]);
			}
			break;
#endif
		  default:
			break;
		}
	}
}

/*================================================================*/
void print_DIMACS_std (FILE * Output, network * n, char * prob_name)
{
	int i, vnum, e_num;
	edge * e;

	vnum = n->vertnum;
	e_num = n->edgenum;
	if (Output == NULL)
	  return;
	
	fprintf(Output, "c --------------------------------------------\n");
	fprintf(Output, "c Solution file for the problem: %s\n", prob_name);
	fprintf(Output, "c Created by the program PLED\n");
	fprintf(Output, "c (c) Tamas Badics 1991\n");
	fprintf(Output, "c --------------------------------------------\n");
	fprintf(Output, "c Parameters of the network:\n");
	fprintf(Output, "c Vertex number:%6d Edge number: %10d\n", vnum, e_num);
	fprintf(Output, "c Source: %6d Sink: %6d\n"
		   , n->source->label, n->sink->label);
	fprintf(Output, "c --------------------------------------------\n");
	if (n->maxflow <= 0){ 
		fprintf(Output, "c NO FEASIBLE SOLUTION\n");
		exit(0);
	}else{ 
		fprintf(Output, "c The maximum flow value:\n");
		fprintf(Output, "s %d\n", (int)n->maxflow);
	}
	
	fprintf(Output, "c --------------------------------------------\n");
	fprintf(Output, "c Flow values of the edges:\n");
	fprintf(Output, "c   FROM     TO       FLOW\n");
	fprintf(Output, "c --------------------------------------------\n");

	for (i = 1; i <= e_num; i++){
		e = &n->edges[i];
		fprintf(Output, "f %6d %6d %10d\n"
			   , e->from->index, e->to->index, (int)e->flow); 
	}
}
/*===============================================================*/

#ifdef DEBUG
void print_edge(edge * e, int quiet)
{
	PRINT( quiet, ("Edge (%6d %6d) Fl:%10d\tCap: %10d\n"
		   ,e->from->index, e->to->index, e->flow, e->cap));
}

void print_vertex(vertex * v, int quiet)
{
	PRINT( quiet, ("Vert:[%6d] Lab: %6d Exc: %10d"
		   , v->index, v->label, v->excess));

	if (v->current){
		PRINT( quiet, (" Cur-"));
		print_edge(*v->current);
	}else
	  PRINT( quiet, (" Cur-Edge: NULL\n"));
}
#endif

/*=================================================================*/
void print_err(char * s, int stop)
{
	fprintf(stderr, "%s\n", s);
	if (stop)
	  exit(-1);
}

