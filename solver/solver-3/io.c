/* io.c */

#include "graph.h"

/* File Format

DIMACS format

c  Comment lines
p Problem Nodes Arcs   -----  problem: min, max, or asn
n id s/t               -----  source and sink designation
a src dst cap  

nodes in range 1 . . n


Solution graphs
c Comment lines
s Solution
f src dst flow

- note: internally, nodes are in the range 0 . . n-1

*/


Graph *InputFlowGraph(f, s, t)
FILE *f;
int *s, *t;
{
  char c, c1, ReadChar(), buff[100];
  Graph *G;
  int i, PType, nodes, edges, v, w, cap;

  G = (Graph *) malloc(sizeof(Graph));
  InitGraph(G);

  PType = UNDEFINED;
  
  while (1) {
    if (EOF_Test(f))
      break;
    c = ReadChar(f);
    switch (c) {

    case 'a':
      v = GetInt(f);
      w = GetInt(f);
      cap = GetInt(f);
      AddEdge(v - 1, w - 1, cap, G);
      break;

    case 'c':
      SkipLine(f);
      break;

    case 'n':
      if (PType == MAXFLOW){
	v = GetInt(f);
	c1 = ReadChar(f);
	if (c1 == 's')
	  *s = v - 1;
	else if (c1 == 't')
          *t = v - 1;
	else
	  Barf("Unexpected node type");
      }
      else {
	Barf("Unimplemented or undefined problem type");
      }
      break;

    case 'p':
      GetString(f, buff);
      if (Strcmp(buff, "max")){
	PType = MAXFLOW;
      }
      else
	Barf("Undefined problem type");
      nodes = GetInt(f);
      edges = GetInt(f);
      break;

    default:
      Barf("Unexpected case in InputFlowGraph\n");
      break;

    }
    
  }

  for (i = 0; i < nodes; i++)
    AddVertex(i, G);

  return G;
}


Graph *InputFlow(f, G, s)
FILE *f;
Graph *G;
int *s;
{
  char c, c1, ReadChar(), buff[100];
  int i, PType, nodes, edges, v, w, flow;
  Edge *e, *EdgeLookup();

  while (1) {
    if (EOF_Test(f))
      break;
    c = ReadChar(f);
    switch (c) {


    case 'c':
      SkipLine(f);
      break;

    case 'f':
      v = GetInt(f);
      w = GetInt(f);
      flow = GetInt(f);
      e = EdgeLookup(v-1,w-1,G);
      if (e == (Edge *) 0)
	Barf("Edge missing from graph");
      e->f += flow;
      e->mate -= flow;
      break;

    case 's':
      *s = GetInt(f);
      break;

    default:
      Barf("Unexpected case in InputFlow\n");
      break;

    }
    
  }

  return G;
}


int GraphOutput(G, f, s, t)
Graph *G;
int s, t;
FILE *f;
{
  int i;

  fprintf(f, "p max %d %d\n", G->size, EdgeCount(G));
  fprintf(f, "n %d s\n", s + 1);
  fprintf(f, "n %d t\n", t + 1);
  for (i = 0; i <= G->max_v; i++){
    WriteVertex(i, G, f);
  }

}


int OutputFlow(G, f, s)
Graph *G;
int s;
FILE *f;
{
  int i;

  fprintf(f, "s %d\n", s);
  for (i = 0; i <= G->max_v; i++){
    WriteVertex2(i, G, f);
  }

}

int PrintFlow(G, s)
Graph *G;
int s;
{
  OutputFlow(G, stdout, s);
}

PrintGraph(G)
Graph *G;
{
  int i;

  for (i = 0; i <= G->max_v; i++){
    if (G->V[i] == FALSE)
      continue;
    WriteVertex3(i, G, stdout);
  }
}


/* for file output */
int WriteVertex(v, G, f)
int v;
Graph *G;
FILE *f;
{
  Edge *e;

  e = G->A[v];
  while (e != (Edge *) 0){
    if (e->c > 0){
        fprintf(f, "a %d %d %d\n", e->t + 1, e->h + 1, e->c);
    }
    e = e->next;
  }
}


int WriteVertex2(v, G, f)
int v;
Graph *G;
FILE *f;
{
  Edge *e;

  e = G->A[v];
  while (e != (Edge *) 0){
    if (e->f > 0){
        fprintf(f, "f %d %d %d\n", e->t + 1, e->h + 1, e->f);
    }
    e = e->next;
  }
}

int WriteVertex3(v, G, f)
int v;
Graph *G;
FILE *f;
{
  Edge *e;

  e = G->A[v];
  while (e != (Edge *) 0){
/*    if (e->c > 0){
        fprintf(f, "%d %d %d %d\n", e->t + 1, e->h + 1, e->c, e->f);
    } */
    fprintf(f, "%d %d %d %d\n", e->t + 1, e->h + 1, e->c, e->f);
    e = e->next;
  }
}


