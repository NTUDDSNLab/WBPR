#include "graph.h"
#include "netflow.h"

main(argc, argv)
int argc;
char *argv[];
{
  Graph *G, *InputFlowGraph();
  FILE *f1, *f2, *OpenFile();
  int  s, t, output, flow, runtime, flowfct;

  if (argc < 2 || argc  > 3)
    Barf("Usage: flow flowfct input (output)");

  if ((f1 = OpenFile(argv[1])) == NULL)
    Barf("Could not open input file");

  if (argc == 3){
    output = TRUE;
    if ((f2 = fopen(argv[2], "w")) == NULL)
      Barf("Could not open output file");
  }
  else
    output = FALSE;

  G = InputFlowGraph(f1, &s, &t);

  if (s < 0 || t < 0 || s >= G->size || t >= G->size)
    Barf("Source or sink out of range");

  BeginTiming();

  flow =  FindFlow(G, s, t, GOLDBERG_1);

  runtime = EndTiming();

  if (output){
    OutputFlow(G, f2, flow);
    fclose(f2);
  }

  ValidFlow(G, s, t);
  printf("Flow value %d\n Runtime %f seconds\n", flow, runtime/100.0);
  printf("Graph: %s, vertices %d, edges %d\n", argv[2], G->size, G->edge_count);
/*  PrintCut(G, s); */
}


int FindFlow(G, s, t, fct)
Graph *G;
int s, t, fct;
{
  int flag, count;
  Graph *L, *CopyGraph();

  flag = TRUE;

  InitFlow(G);
 
  if (fct == GOLDBERG_1 || fct == GOLDBERG_2 || 
      fct == GOLDBERG_3 || fct == GOLDBERG_4 || fct == GOLDBERG_5){
    Goldberg(G, s, t, fct);
    return VertexFlow(s, G);
  }

  switch (fct){
  case DINIC:
  case KARZANOV:
    L = CopyGraph(G);
    break;
  default:
    L = (Graph *) 0;
    break;
  } 
  
  count = 0;

  while(flag){
    flag = AugmentFlow(G, s, t, fct, L);
    count++;
  }

  printf("Augmentations: %d\n",count);
  return VertexFlow(s, G);
}

int VertexFlow(i, G)
int i;
Graph *G;
{
  Edge *e;
  int flow;

  flow = 0;
  e = G->A[i];
  while (e != (Edge *) 0){
    flow += e->f;
    e = e->next;
  }

  return flow;
}

ValidFlow(G, s, t)
Graph *G;
int s, t;
{
  int i;
  Edge *e;

  for (i = 0; i < G->size; i++){
    e = G->A[i];
    while (e != (Edge *) 0){
      if (e->f != -e->mate->f)
	Barf("Antisymmetry violated");
      if (e->f > e->c)
	Barf("Capacity violated");
      e = e->next;
    }
  }
  for (i = 0; i< G->size; i++){
    if (i == s || i == t)
      continue;
    if (VertexFlow(i, G) != 0)
      Barf("Conservation violated");
  }
  if (VertexFlow(s, G) != -VertexFlow(t, G))
    Barf("Network leaks!");

}

MarkCut(G, u, C, n)
Graph *G;
int u, C[], *n;
{
  int M[MAX_N], S[MAX_N], h, t, i, v, count;
  Edge *e;

  for (i = 0; i < G->size; i++)
    C[i] = 0;

  count = 1;
  h = t = 0;
  S[0] = u;
  C[u] = u;
  while (t <= h){
    v = S[t++];
    e = G->A[v];
    while (e != (Edge *) 0){
      if (C[e->h] == 0 && e->f < e->c){
	count++;
	C[e->h] = 1;
	S[++h] = e->h;
      }
      e = e->next;
    }
  }
  *n = count;
}

PrintCut(G, s)
Graph *G;
int s;
{
  int C[MAX_N], n, i, count;
  Edge *e;

  MarkCut(G, s, C, &n);
  printf("Reachable from source %d\n", n);

  printf("Cut vertices:\n");
  
  count = 0;
  for (i = 0; i < G->size; i++){
    if (C[i] == 0)
      continue;
    e = G->A[i];
    while (e != (Edge *) 0){
      if (C[e->h] == 0){
	printf("%d ", i);
	if (++count % 10 == 0)
	  printf("\n");
	break;
      }
      e = e->next;
    }
  }
  printf("\n");
}

int AugmentFlow(G, s, t, fct, L)
Graph *G, *L;
int s, t, fct;
{
  int P[MAX_N], flag;

 
  switch (fct) {
  case DFS:
    if ((flag = FindPath1(G, s, t, P)) == TRUE)
      AddPath(G, t, P);
    break;
  case BFS:
    if ((flag = FindPath2(G, s, t, P)) == TRUE)
      AddPath(G, t, P);
    break;
  case MAX_GAIN:
    if ((flag = FindPath3(G, s, t, P)) == TRUE)
      AddPath(G, t, P);
    break;
  case DINIC:
  case KARZANOV:
    if ((flag = LayeredGraph(G, s, t ,L)) == TRUE){
      BlockingFlow(L, s, t, fct);
      AddFlow(G, L);
    }
    break;
  case DINIC_NEW:
    if ((flag = LG2(G, s, t)) == TRUE){
      BlockingFlow(G, s, t, fct);
    }
    break;
  default:
    Barf("Unknown case in augment flow");
    break;
  }

  return flag;
}

AddPath(G, t, P)
Graph *G;
int *P, t;
{
  Edge *e, *EdgeLookup();
  int i, b; 

  i = 0;
  b = MAX_CAP;

  while (P[i] != t){
    e = EdgeLookup(P[i], P[i+1], G);
    b = (e->c - e->f < b) ? e->c - e->f : b;    
    i++;
  }

  i = 0;
  while (P[i] != t){
    e = EdgeLookup(P[i], P[i+1], G);
    e->f += b;
    e->mate->f -= b;
    i++;
  }
}

PrintAugment(P, t, b)
int P[], t, b;
{
  int i;

  printf("Augmentation %d\n", b);
  i = 0;
  do {
    printf("%d ", P[i]);
  }
  while (P[i++] != t);
  printf("\n");

}



InitFlow(G)
Graph *G;
{
  int i;
  Edge *e;

  for (i = 0; i < G->size; i++){
    e = G->A[i];
    while (e != (Edge *) 0){
      e->f = 0;
      e = e->next;
    }
  }

}


