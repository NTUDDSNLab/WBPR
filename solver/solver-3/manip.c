/* manip.c */
#include "graph.h"

InitGraph(G)
Graph *G;
{
  int i;

  for (i = 0; i < MAX_N; i++){
    G->A[i] = (Edge *) 0;
    G->V[i] = FALSE;
  }
  G->size = 0;  
  G->max_v = -1;
  G->edge_count = 0;
}

Graph *CopyGraph(G1)
Graph *G1;
{
  int i;
  Edge *e;
  Graph *G2;

  G2 = (Graph *) malloc(sizeof(Graph));
  InitGraph(G2);

  for (i = 0; i <= G1->max_v; i++){
    if (G1->V[i] == TRUE){
      AddVertex(i, G2);
      e = G1->A[i];
      while (e != (Edge *) 0){
	if (e->c > 0)
	  AddEdge(i, e->h, e->c, G2);
	e = e->next;
      }
    }
  }

  return G2;
}

AddVertex(v, G)
int v;
Graph *G;
{
  if (G->V[v] == TRUE)
    Barf("Vertex already present");

  G->V[v] = TRUE;
  G->size++;
  if (v > G->max_v)
    G->max_v = v;
}


AddEdge(v1, v2, a, G)
int v1, v2, a;
Graph *G;
{
  Edge *e1, *e2, *EdgeLookup();

  if (v1 == v2)
    Barf("No Loops");

  if ((e1 = EdgeLookup(v1, v2, G)) != (Edge *) 0){
    e1->c += a;
    return;
  }

  e1 = (Edge *) malloc(sizeof(Edge));
  e2 = (Edge *) malloc(sizeof(Edge));

  e1->mate = e2;
  e2->mate = e1;

  e1->next = G->A[v1];
  G->A[v1] = e1;
  e1->t = v1;
  e1->h = v2;
  e1->c = a;

  e2->next = G->A[v2];
  G->A[v2] = e2;
  e2->t = v2;
  e2->h = v1;
  e2->c = 0;

  G->edge_count++;
}

Edge *EdgeLookup(v1, v2, G)
int v1, v2;
Graph *G;
{
  Edge *e;

  e = G->A[v1];
  while (e != (Edge *) 0){
    if (e->h == v2)
      return e;
    e = e->next;
  }
  return (Edge *) 0;
}

UEdgeArray(E, m, G)
Edge *E[];
int m;
Graph *G;
{
  int i, count;
  Edge *e;

  count = 0;
  for (i = 0; i <= G->max_v; i++){
    if (G->V[i] == FALSE)
      continue;
    e = G->A[i];
    while (e != (Edge *) 0){
      if (e->h < e->t){
	if (count == m)
	  Barf("UEdgeArray overflow");
        E[count] = e;
	count++;
      }
      e = e->next;
    }
  }
}

/* Count the number of edges with positive capacity */
int EdgeCount(G)
Graph *G;
{
  int i, count;
  Edge *e;

  count = 0;
  for (i = 0; i <= G->max_v; i++){
    if (G->V[i] == FALSE)
      continue;
    e = G->A[i];
    while (e != (Edge *) 0){
      if (e->c > 0)
	count++;
      e = e->next;
    }
  }
  return count;
}
  


