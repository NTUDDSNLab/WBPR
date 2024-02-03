/* dinic.c */

#include "graph.h"
#include "netflow.h"

int LayeredGraph(G, s, sink, L)
Graph *G, *L;
int s, sink;
{
  int M[MAX_N], S[MAX_N], h, t, i, v, r;
  Edge *e, *e1, *EdgeLookup();

  for (i = 0; i < G->size; i++)
    M[i] = -1;

  h = t = 0;
  S[0] = s;
  M[s] = 0;
  while (h >= t){
    v = S[t++];

    e = G->A[v];
    while (e != (Edge *) 0){
      r = e->c - e->f;
      e1 = EdgeLookup(v, e->h, L);
      if (r > 0){
	if (M[e->h] == -1){
	  M[e->h] = M[v] + 1;
	  S[++h] = e->h;
	  e1->c = r;
	}
	else if (M[e->h] == M[v] + 1)
	  e1->c = r;
	else
	  e1->c = 0;	
      }
      else
	e1->c = 0;
      e = e->next;
    }
  }

  InitFlow(L);  

  return M[sink] != -1;

}

int LG2(G, s, sink)
Graph *G;
int s, sink;
{
  int M[MAX_N], S[MAX_N], h, t, i, v, r;
  Edge *e;

  for (i = 0; i < G->size; i++){
    e = G->A[i];
    while (e != (Edge *) 0){
      e->flag = FALSE;
      e = e->next;
    }
    M[i] = -1;
  }

  h = t = 0;
  S[0] = s;
  M[s] = 0;
  while (h >= t){
    v = S[t++];
    e = G->A[v];
    while (e != (Edge *) 0){
      r = e->c - e->f;
      if (r > 0){
	if (M[e->h] == -1){
	  M[e->h] = M[v] + 1;
	  S[++h] = e->h;
	  e->flag = TRUE;
	}
	else if (M[e->h] == M[v] + 1)
	  e->flag = TRUE;
      }
      e = e->next;
    }
  }

  return M[sink] != -1;

}

int BlockingFlow(L, s, t, fct)
Graph *L;
int s, t, fct;
{
  int P[MAX_N];

  switch (fct) {
  case DINIC:
    while (FindPath4(L, s, t, P))
      AddPath(L, t, P);
    break;
  case DINIC_NEW:
    Dinic(L, s, t);
    break;
  case KARZANOV:
    Karzanov(L, s, t);
    break;
  default:
    Barf("Unexpected case in blocking flow");
    break;
  }
}

AddFlow(G, L)
Graph *G, *L;
{
  int i;
  Edge *e, *e1, *EdgeLookup();

  for (i = 0; i < L->size; i++){
    e = L->A[i];
    while (e != (Edge *) 0){
      if (e->f > 0){
	e1 = EdgeLookup(i, e->h, G);
	e1->f += e->f;
	e1->mate->f -= e->f;
      }
      e = e->next;
    }
  }
    
}


int FindPath4(G, v1, v2, P)
Graph *G;
int v1, v2, P[];
{
  int M[MAX_N], S[MAX_N], sp, i, v, d;
  Edge *e;


  for (i = 0; i < G->size; i++)
    M[i] = -1;

  sp = 0;
  S[0] = v1;
  M[v1] = v1;
  while (M[v2] == -1){
    if (sp < 0)
      return FALSE;
    v = S[sp--];

    e = G->A[v];
    while (e != (Edge *) 0){
      if (M[e->h] == -1 && e->f < e->c && e->c > 0){
	M[e->h] = v;
	S[++sp] = e->h;
      }
      e = e->next;
    }
  }

  d = 1;
  v = v2;
  while (M[v] != v1){
    d++;
    v = M[v];
  }

  v = v2;
  while (d >= 0){
    P[d] = v;
    v = M[v];
    d--;
  }
  return TRUE;
}


Dinic(G, v1, v2)
Graph *G;
int v1, v2;
{
  int done, sp, v, i, S[MAX_N], flow;
  Edge *current[MAX_N], *e;

  for (i = 0; i < G->size; i++)
    current[i] = G->A[i];

  flow = 0;
  sp = 0;
  v = S[0] = v1;
  done = FALSE;

  while (done == FALSE){
    e = current[v];
    while (e != (Edge *) 0 && (e->f == e->c || e->flag == FALSE)){
      e = current[v] = e->next;
    }
    if (e == (Edge *) 0){
      if (v == v1)
	done = TRUE;
      else {
	v = S[--sp];
	current[v] = current[v]->next;
      }
    }
    else {
      S[++sp] = v = e->h;
      if (v == v2){
	flow += UpdatePath(G, S, sp, current);
	v = v1;
	sp = 0;
      }
    }
  }

}

int UpdatePath(G, S, n, current)
Graph *G;
int S[], n;
Edge *current[];
{
  Edge *e;
  int i, b; 

  i = 0;
  b = MAX_CAP;

  for (i = 0; i < n; i++){
    e = current[S[i]];
    b = (e->c - e->f < b) ? e->c - e->f : b;    
  }

  for (i = 0; i < n; i++){
    e = current[S[i]];
    e->f += b;
    e->mate->f -= b;
  }
  return b;
}

FindPath5(G, s, t, P)
Graph *G;
int s, t, P[];
{
  int i, v, w, r, d;

  Edge *e;

  Heap.size = 0;
  Heap.range = G->max_v+1;

  for (i = 0; i < G->max_v+1; i++)
    Heap.hptr[i] = Heap.a[i] = Heap.D[i] = 0;

  for (i = 0; i <= G->max_v; i++){
    Insert(i, 0);
    Pred[i] = -1;
  }

  IncreaseKey(s, INFINITY);

  while (Heap.size > 0){


    w = DeleteMax();
    e = G->A[w];
    while (e != (Edge *) 0){
      v = e->h;
      r = e->c - e->f;
      if (Heap.hptr[v] >= 0 && r > Heap.D[v] && e->c > 0){
	IncreaseKey(v, Min(r, Heap.D[w]));
	Pred[v] = w;
      }
      e = e->next;
    }
  }


  if (Heap.D[t] <= 0)
    return FALSE;

  d = 1;
  v = t;
  while (Pred[v] != s){
    d++;
    v = Pred[v];
  }

  v = t;;
  while (d >= 0){
    P[d] = v;
    v = Pred[v];
    d--;
  }
  return TRUE;

}


Karzanov(L, s, t)
Graph *L;
int s, t;
{
  int A[MAX_N], n, flag, i, delta[MAX_N], block[MAX_N], v, r;
  Edge *e, *outedge[MAX_N], *inedge[MAX_N];

  TopoSort(L, A, s, t, &n);

  for (i = 0; i < L->size; i++){
    delta[i] = block[i] = 0;
    outedge[i] = inedge[i] = L->A[i];
  }


  e = L->A[s];
  while (e != (Edge *) 0){
    if (e->c > 0){
      e->f = e->c;
      e->mate->f = -e->f;
      delta[e->h] += e->f;
    }
    e = e->next;
  }

  flag = 1;
  while (flag > 0){
    for (i = 0; i < n; i++){         /* Increasing Step */
      v = A[i];
      if (v != s && v != t && delta[v] > 0 && block[v] == 0){
	e = outedge[v];
	while (e != (Edge *) 0){
	  r = Min(e->c - e->f, delta[v]);
	  if (e->c > 0 && r > 0 && block[e->h] == 0){
	    e->f += r;
	    e->mate->f -= r;
	    delta[v] -= r;
	    delta[e->h] += r;
	  }
	  if (delta[v] == 0)
	    break;
	  e = e->next;
	}
	if (delta[v] > 0)
	  block[v] = 1;
	outedge[v] = e;

      }
    }

    for (i = n-1; i >= 0; i--){         /* Decreasing Step */
      v = A[i];
      if (v != s && v != t && delta[v] > 0 && block[v] == 1){
	e = inedge[v];
	while (e != (Edge *) 0){
	  r = Min(e->mate->f, delta[v]);
	  if (r > 0){
	    e->f += r;
	    e->mate->f -= r;
	    delta[v] -= r;
	    delta[e->h] += r;
	  }
	  if (delta[v] == 0)
	    break;
	  e = e->next;
	}
	inedge[v] = e;
      }
    }

    flag = 0;
    for (i = 0; i < n; i++){
      v = A[i];
      if (v != s && v != t && delta[v] > 0){
	flag = 1;
	break;
      }
    }
  }
}


/* Topological Sort of a graph, of the vertices reachable from s.  The 
only edges considered are the edges of positive capacity.  Results stored
in matrix A, with the number of vertices returned in n.
*/
TopoSort(G, A, s, t, n)
Graph *G;
int *A, s, t, *n;
{
  int degree[MAX_N], i, stack[MAX_N], sptr, count, v;
  Edge *e;

  Prune(G, s, t);

  for (i = 0; i < G->size; i++){
    degree[i] = 0;
    e = G->A[i];
    while (e != (Edge *) 0){
      if (e->mate->c > 0)
	degree[i]++;
      e = e->next;
    }
  }

  sptr = 0;
  for (i = 0; i < G->size; i++)
    if (degree[i] == 0)
      stack[sptr++] = i;

  count = 0;
  while (sptr > 0){
    v = stack[--sptr];
    A[count++] = v;
    e = G->A[v];
    while (e != (Edge *) 0){
      if (e->c > 0){
	degree[e->h]--;
	if (degree[e->h] == 0)
	  stack[sptr++] = e->h;
      }
      e = e->next;
    }
  }

  *n = count;
}


/* Prune unreachable vertices - this is done by setting capacities 
to zero */
Prune(G, s, t)
Graph *G;
int s, t;
{
  int i, a[MAX_N], b[MAX_N], queue[MAX_N], q1, q2, v;
  Edge *e, *e1;

  for (i = 0; i < G->size; i++)
    a[i] = b[i] = 0;

  q1 = q2 = 0;             /* Forward Search from s */
  queue[0] = s;
  a[s] = 1;
  while (q2 <= q1){
    v = queue[q2++];

    e = G->A[v];
    while (e != (Edge *) 0){
      if (a[e->h] == 0 && e->f < e->c){
	a[e->h] = 1;
	queue[++q1] = e->h;
      }
      e = e->next;
    }
  }

  q1 = q2 = 0;             /* Backward Search from t */
  queue[0] = t;
  b[t] = 1;
  while (q2 <= q1){
    v = queue[q2++];

    e = G->A[v];
    while (e != (Edge *) 0){
      e1 = e->mate;
      if (b[e1->t] == 0 && e1->f < e1->c){
	b[e1->t] = 1;
	queue[++q1] = e1->t;
      }
      e = e->next;
    }
  }

  for (i = 0; i < G->size; i++)
    if (a[i] == 0 || b[i] == 0){
      e = G->A[i];
      while (e != (Edge *) 0){
	e->c = e->mate->c = 0;
	e = e->next;
      }
    }
}





















