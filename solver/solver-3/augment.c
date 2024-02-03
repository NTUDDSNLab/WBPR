/*augment.c */

#include "graph.h"
#include "netflow.h"

/* Find a single augmenting path */
int FindPath1(G, v1, v2, P)
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
      if (M[e->h] == -1 && e->f < e->c){
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


/* Breadth first search */
int FindPath2(G, v1, v2, P)
Graph *G;
int v1, v2, P[];
{
  int M[MAX_N], S[MAX_N], h, t, i, v, d;
  Edge *e;

  for (i = 0; i < G->size; i++)
    M[i] = -1;

  h = t = 0;
  S[0] = v1;
  M[v1] = v1;
  while (M[v2] == -1){
    if (t > h)
      return FALSE;
    v = S[t++];

    e = G->A[v];
    while (e != (Edge *) 0){
      if (M[e->h] == -1 && e->f < e->c){
	M[e->h] = v;
	S[++h] = e->h;
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



FindPath3(G, s, t, P)
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
      if (Heap.hptr[v] >= 0 && r > Heap.D[v]){
	IncreaseKey(v, Min(r, Heap.D[w]));
	Pred[v] = w;
      }
      e = e->next;
    }
  }

/*  Verify(G, s);

  PrintResult(s, t);
*/
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

int DeleteMax()
{
  int i, j, t, temp;

  if (Heap.size == 0)
    Barf("Attempt to delete from empty heap");

  t = Heap.a[0];
  Heap.hptr[t] = -2;

  Heap.size--;

  if (Heap.size == 0)
    return t;

  Heap.a[0] = Heap.a[Heap.size];
  Heap.hptr[Heap.a[0]] = 0;

  i = 0;
  while (2*i + 2 <= Heap.size){
    j = ((2*i+2==Heap.size) || (Heap.D[Heap.a[2*i+1]]>Heap.D[Heap.a[2*i+2]])) ?
           2*i+1 : 2*i + 2;
    if (Heap.D[Heap.a[i]] > Heap.D[Heap.a[j]])
      break;
    temp = Heap.a[i];  Heap.a[i] = Heap.a[j];  Heap.hptr[Heap.a[i]] = i;
    Heap.a[j] = temp; Heap.hptr[Heap.a[j]] = j;
    i = j;
  }
  return t;
}

HeapTest()
{
  int i, j;

  for (i = 0; i < Heap.size; i++){
    j = 2*i+1;
    if (j < Heap.size && Heap.D[Heap.a[i]] < Heap.D[Heap.a[j]])
      Barf("Heap Condition violated");
    j = 2*i + 2;
    if (j < Heap.size && Heap.D[Heap.a[i]] < Heap.D[Heap.a[j]])
      Barf("Heap Condition violated");
  }
}


IncreaseKey(v, k)
int v, k;
{
  int i, j, temp;

  if (Heap.hptr[v] < 0 || Heap.hptr[v] >= Heap.size)
    Barf("Not in Heap");
  if (Heap.D[v] > k)
    Barf("Attempt to decrease key in increase key");

  Heap.D[v] = k;
  i = Heap.hptr[v];

  while (i > 0){
    j = (i - 1) / 2;
    if (Heap.D[Heap.a[i]] <= Heap.D[Heap.a[j]])
      break;
    Heap.hptr[Heap.a[j]] = i;
    temp = Heap.a[i];  Heap.a[i] = Heap.a[j]; Heap.a[j] = temp;
    i = j;
  }
  Heap.hptr[v] = i;

}

Insert(v, k)
int v, k;
{
  int i, j, temp;

  if (Heap.size == Heap.range)
    Barf("Heap overflow");

  i = Heap.size;
  Heap.size++;

  Heap.D[v] = k;

  Heap.a[i] = v;
  while (i > 0){
    j = (i - 1) / 2;
    if (Heap.D[Heap.a[i]] >= Heap.D[Heap.a[j]])
      break;
    Heap.hptr[Heap.a[j]] = i;
    temp = Heap.a[i];  Heap.a[i] = Heap.a[j]; Heap.a[j] = temp;
    i = j;
  }
  Heap.hptr[v] = i;
}



Verify(G,s)
Graph *G;
int s;
{
  int i, v, cost;
  Edge *e;

  for (i = 0; i <= G->max_v; i++){
    if (i == s){
      if (Pred[i] != -1 || Heap.D[i] != INFINITY)
	Barf("Verification Error");
    }
    else {
      v = Pred[i];
      if (v == -1)
	continue;
      if (v < 0 || v > G->max_v)
	Barf("Verification Error");
      e = G->A[v];
      cost = -1;
      while (e != (Edge *) 0){
	if (e->h == i){
	  cost = e->c - e->f;
	  break;
	}
	e = e->next;
      }
      if (cost < 0)
	Barf("Verification Error");
      if (Min(cost,Heap.D[v]) != Heap.D[i])
	Barf("Verification Error");
    }

  }
  printf("Verification Successful\n");
}

PrintResult(s,t)
int s, t;
{
  int i, v;

  printf("The shortest path from %d to %d has length %d\n", s, t, Heap.D[t]);
  printf("The path is:\n");
  i = 0;
  v = t;
  do {
    printf("%d ", v);
    v = Pred[v];
    i++;
    if (i % 13 == 0)
      printf("\n");
  } while (v != -1);
  printf("\n");
}










