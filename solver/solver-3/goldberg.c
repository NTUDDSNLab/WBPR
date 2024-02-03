/* goldberg.c */

#include "graph.h"
#include "netflow.h"

int Excess[MAX_N], Dist[MAX_N];
int RCount, UCount, SCount;


Edge *Current[MAX_N];

Goldberg(G, s, t, fct)
Graph *G;
int s, t, fct; 
{
  Edge *e;
  Queue *Q, *MakeQueue();
  int v;

  InitGoldberg(G, s);
  InitRandom(0);

  RCount = SCount = UCount = 0;

  switch (fct) {
  case GOLDBERG_1:
/*    SortEdgeLists(G); */
    Goldberg1(G, s, t);
    break;
  case GOLDBERG_2:
    Goldberg2(G, s, t);
    break;
  case GOLDBERG_3:
    Goldberg3(G, s, t);
    break;
  case GOLDBERG_4:
    Goldberg4(G, s, t);
    break;
  case GOLDBERG_5:
    Goldberg5(G, s, t);
    break;
  default:
    Barf("Impossible Case");
    break;
  }

  printf("SCount %d UCount %d RCount %d\n", 
	  SCount, UCount, RCount);
}

Goldberg1(G, s, t)
Graph *G;
int s, t;
{
  Edge *e, *EdgeLookup();
  Queue *Q, *MakeQueue();
  int v;
  int count, checkpoint, LabelCount;
  int State, b1, b2, b3, b4, b5;
  int oldex = 0;

  SetLabels(G, t, s); 

  Q = MakeQueue(G->size);
  e = G->A[s];
  while (e != (Edge *) 0){
    if (e->c > 0 && e->h != t)
      Enqueue(Q, e->h);
    e = e->next;
  }
  count = 0;   checkpoint = 0;   LabelCount = 0;

  while (! QueueEmpty(Q)){
    count++;
    checkpoint++;
    if (count >= G->edge_count/2){ 
      count = 0;
      LabelCount++;
      SetLabels(G, t, s);   
    }
    v = Dequeue(Q);
    Discharge(v, G,  Q, s, t);
    if (Excess[v] > 0 && Dist[v] < INFINITY)
      Enqueue(Q, v);
  }
  printf("Relabelings :%d\n", LabelCount);
}

Goldberg2(G, s, t)
Graph *G;
int s, t;
{
  Edge *e;
  IHeap *IH, *MakeIHeap();
  int v;
  int count, checkpoint, LabelCount;

  SetLabels(G, t, s); 

  IH = MakeIHeap(2 * G->size);
  e = G->A[s];
  while (e != (Edge *) 0){
    if (e->c > 0 && e->h != t)
      IHInsert(IH, e->h, Dist[e->h]);
    e = e->next;
  }

  count = 0;   checkpoint = 0;   LabelCount = 0;
  while (! IHEmpty(IH)){
    count++;
    checkpoint++;
/*    if (checkpoint % (G->edge_count / 5) == 0)
      printf("%d\n", IH->elts); */
    if (count >= G->edge_count/2){ 
      count = 0;
      LabelCount++;
      SetLabels(G, t, s); 
    }    
    v = IHDeleteMin(IH);
/*printf("%d:Excess %d, Level %d\n", v+1, Excess[v], Dist[v]);*/
    Discharge2(v, G,  IH, s, t);
    if (Excess[v] > 0 && Dist[v] < INFINITY)
      IHInsert(IH, v, Dist[v]);
  }
  printf("Relabelings :%d\n", LabelCount);
}

Goldberg3(G, s, t)
Graph *G;
int s, t;
{
  Edge *e;
  Queue *Q, *MakeQueue();
  int v;
  int count, checkpoint, LabelCount, i;

  Heap.size = 0;
  Heap.range = G->max_v+1;

  for (i = 0; i < G->max_v+1; i++)
    Heap.hptr[i] = Heap.a[i] = Heap.D[i] = 0;

  SetLabels(G, t, s); 

  e = G->A[s];
  while (e != (Edge *) 0){
    if (e->c > 0 && e->h != t)
      Insert(e->h, e->f);
    e = e->next;
  }
  count = 0;   checkpoint = 0;   LabelCount = 0;

  while (Heap.size > 0){
    count++;
    checkpoint++;
    if (checkpoint % (G->edge_count / 5) == 0)
      printf("%d\n", Heap.size); 
    if (count >= G->edge_count / 2){ 
      count = 0;
      LabelCount++;
      SetLabels(G, t, s); 
    }
    v = DeleteMax();
/* printf("%d:Excess %d, Level %d\n", v+1, Excess[v], Dist[v]); */
/*    printf("%d: %d\n", v, Excess[v]); */
    Discharge3(v, G, s, t);
    if (Excess[v] > 0 && Dist[v] < INFINITY)
      Insert(v, Excess[v]);
  }
  printf("Relabelings :%d\n", LabelCount);
}

Goldberg4(G, s, t)
Graph *G;
int s, t;
{
  Edge *e;
  Stack *Q, *MakeStack();
  int v;
  int count, checkpoint, LabelCount;


  SetLabels(G, t, s); 

  Q = MakeStack(G->size);
  e = G->A[s];
  while (e != (Edge *) 0){
    if (e->c > 0 && e->h != t)
      StackPush(Q, e->h);
    e = e->next;
  }
  count = 0;   checkpoint = 0;   LabelCount = 0;

  while (! StackEmpty(Q)){
    count++;
    checkpoint++;
    if (checkpoint % (G->edge_count / 5) == 0)
      printf("%d\n", StackSize(Q)); 
    if (count >= G->edge_count/2){ 
      count = 0;
      LabelCount++;
      SetLabels(G, t, s); 
    }
    v = StackPop(Q);
/*printf("%d:Excess %d, Level %d\n", v+1, Excess[v], Dist[v]);*/
    Discharge4(v, G,  Q, s, t);

    if (Excess[v] > 0 && Dist[v] < INFINITY)
      StackPush(Q, v);
  }
  printf("Relabelings :%d\n", LabelCount);
}


PushRelabel(v, G, Q, s, t)
int v, s, t;
Graph *G;
Queue *Q;
{
  Edge *e;

  e = Current[v];
  if (e->c > e->f  && Dist[v] == Dist[e->h] + 1){
    if (Excess[e->h] == 0 && e->h != s && e->h != t)
      Enqueue(Q, e->h);
    Push(e);
  }
  else if (e->next == (Edge *) 0){
    Current[v] = G->A[v];
    Relabel(v, G);
  }
  else
    Current[v] = e->next;
}

Discharge(v, G, Q, s, t)
int v, s, t;
Graph *G;
Queue *Q;
{
  int d;

  d = Dist[v];
  while (Dist[v] == d && Excess[v] > 0)
    PushRelabel(v, G, Q, s, t);
}


NewDischarge(v, G, Q, s, t)
int v, s, t;
Graph *G;
Queue *Q;
{
  int d;

  d = Dist[v];
  while (Excess[v] > 0)
    PushRelabel(v, G, Q, s, t);
}

Push(e)
Edge *e;
{
  int v, w, d;

  
  v = e->t;  w = e->h;
  d = Min(e->c - e->f, Excess[v]);

  if (d == e->c - e->f)
    SCount++;
  else
    UCount++;
    
  e->f += d;
  e->mate->f -= d;
  Excess[v] -= d;
  Excess[w] += d;
}

Relabel(v, G)
int v;
Graph *G;
{
  int d, r, old;
  Edge *e;

  
  d = INFINITY;
  e = G->A[v];
  while (e != (Edge *) 0){
    r = e->c - e->f;
    if (r > 0 && Dist[e->h] + 1 < d)
      d = Dist[e->h] + 1;
    e = e->next;
  }
  Dist[v] = d;
  RCount++;

}

InitGoldberg(G, s)
Graph *G;
int s;
{
  int i;
  Edge *e;

  for (i = 0; i < G->size; i++){
    Excess[i] = Dist[i] = 0;
    Current[i] = G->A[i];
  }

  Dist[s] = G->size;

  e = G->A[s];
  while (e != (Edge *) 0){
    if (e->c > 0){
      e->f = e->c;
      e->mate->f = -e->c;
      Excess[e->h] += e->c;
    }
    e = e->next;
  }


}




PushRelabel2(v, G, IH, s, t)
int v, s, t;
Graph *G;
IHeap *IH;
{
  Edge *e;

  e = Current[v];
  if (e->c > e->f  && Dist[v] == Dist[e->h] + 1){
    if (Excess[e->h] == 0 && e->h != s && e->h != t)
      IHInsert(IH, e->h, Dist[e->h]);
    Push(e);
  }
  else if (e->next == (Edge *) 0){
    Current[v] = G->A[v];
    Relabel(v, G);
  }
  else
    Current[v] = e->next;
}

Discharge2(v, G, IH, s, t)
int v, s, t;
Graph *G;
IHeap *IH;
{
  int d;

  d = Dist[v];
  while (Dist[v] == d && Excess[v] > 0)
    PushRelabel2(v, G, IH, s, t);
}

Discharge3(v, G,s, t)
int v, s, t;
Graph *G;
{
  int d;

  d = Dist[v];
  while (Dist[v] == d && Excess[v] > 0)
    PushRelabel3(v, G, s, t);
}

PushRelabel3(v, G, s, t)
int v, s, t;
Graph *G;
{
  Edge *e;

  e = Current[v];
  if (e->c > e->f  && Dist[v] == Dist[e->h] + 1){
    if (Excess[e->h] == 0 && e->h != s && e->h != t)
      Insert(e->h, Min(Excess[v], e->c - e->f));
    else if (e->h != s && e->h != t)
      IncreaseKey(e->h, Min(Excess[v], e->c - e->f) + Excess[e->h]);
    Push(e);
  }
  else if (e->next == (Edge *) 0){
    Current[v] = G->A[v];
    Relabel(v, G);
  }
  else
    Current[v] = e->next;
}

Discharge4(v, G, Q, s, t)
int v, s, t;
Graph *G;
Stack *Q;
{
  int d;

  d = Dist[v];
  while (Dist[v] == d && Excess[v] > 0)
    PushRelabel4(v, G, Q, s, t);
}

PushRelabel4(v, G, Q, s, t)
int v, s, t;
Graph *G;
Stack *Q;
{
  Edge *e;

  e = Current[v];
  if (e->c > e->f  && Dist[v] == Dist[e->h] + 1){
    if (Excess[e->h] == 0 && e->h != s && e->h != t)
      StackPush(Q, e->h);
    Push(e);
  }
  else if (e->next == (Edge *) 0){
    Current[v] = G->A[v];
    Relabel(v, G);
  }
  else
    Current[v] = e->next;
}


IHeap *MakeIHeap(n)
int n;
{
  IHeap *IH;
  int i;

  IH = (IHeap *) Alloc(sizeof(IHeap));

  for (i = 0; i < n; i++)
    IH->data[i] = (LNode *) 0;
  IH->freelist = (LNode *) 0;
  IH->max = -1;
  IH->size = n;
  IH->elts = 0;

  return IH;
}

IHDeleteMin(IH)
IHeap *IH;
{
  LNode *l;

  while (IH->data[IH->max] == (LNode *) 0){
    IH->max--;
    if (IH->max == -1)
      break;
  }


  if (IH->max == -1)
    Barf("Attempt to delete from empty IHeap");

  l = IH->data[IH->max];
  IH->data[IH->max] = l->next;
  l->next = IH->freelist;
  IH->freelist = l;
  IH->elts--;

  return l->v;
}


IHInsert(IH, x, k)
IHeap *IH;
int x, k;
{
  LNode *l;

  k %= IH->size;     /* Kludge */

  if (IH->freelist == (LNode *) 0)
    l = (LNode *) Alloc(sizeof(LNode));
  else {
    l = IH->freelist;
    IH->freelist = IH->freelist->next;
   }

  l->v = x;
  l->next = IH->data[k];
  IH->data[k] = l;

  if (k > IH->max)
    IH->max = k;
  IH->elts++;

}

IHEmpty(IH)
IHeap *IH;
{
  while (IH->data[IH->max] == (LNode *) 0){
    IH->max--;
    if (IH->max == -1)
      break;
  }

  return IH->max == -1;
}



Goldberg5(G, s, t)
Graph *G;
int s, t;
{
  Edge *e, *EdgeLookup();
  Queue *Q, *MakeQueue();
  int v;
  int count, checkpoint, LabelCount;
  int State, b1, b2, b3, b4, b5;
  int oldex = 0;

  SetLabels(G, t, s); 

  Q = MakeQueue(G->size);
  e = G->A[s];
  while (e != (Edge *) 0){
    if (e->c > 0 && e->h != t)
      Enqueue(Q, e->h);
    e = e->next;
  }
  count = 0;   checkpoint = 0;   LabelCount = 0;

  while (! QueueEmpty(Q)){
    count++;
    checkpoint++;
    if (count >= G->edge_count/2){ 
      count = 0;
      LabelCount++;
      SL2(G, t, s);
    }

    v = Dequeue(Q);
    NewDischarge(v, G,  Q, s, t);
    if (Excess[v] > 0 && Dist[v] < INFINITY)
      Enqueue(Q, v);
  }
  printf("Relabelings :%d\n", LabelCount);
}







int WriteVertex4(v, G)
int v;
Graph *G;
{
  Edge *e;

  e = G->A[v];
  while (e != (Edge *) 0){
    printf("%d %d %d %d\n", e->t + 1, e->h + 1, e->c, e->f);
    e = e->next;
  }
}

SL2(G, v1, v2)
Graph *G;
int v1, v2;
{
  int  S[MAX_N], LDist[MAX_N], h, t, i, v, count;
  Edge *e, *e1;

  for (i = 0; i < G->size; i++)
      LDist[i] = -1;

  count = 0;
  h = t = 0;
  S[0] = v1;
  LDist[v1] = 0;
  while (t <= h){
    count++;
    v = S[t++];
    e = G->A[v];
    while (e != (Edge *) 0){
      e1 = e->mate;
      if (LDist[e1->t] == -1 && e1->f < e1->c){
	LDist[e1->t] = LDist[v] + 1;
	S[++h] = e1->t;
      }
      e = e->next;
    }
  }
  for (i = 0; i < G->size; i++)
    if (LDist[i] == -1 && Dist[i] < G->size)
      Dist[i] = G->size;

}


SetLabels(G, v1, v2)
Graph *G;
int v1, v2;
{
  int  S[MAX_N], h, t, i, v;
  Edge *e, *e1;

  for (i = 0; i < G->size; i++)
      Dist[i] = -1;

  h = t = 0;
  S[0] = v1;
  Dist[v1] = 0;
  while (t <= h){
    v = S[t++];
    e = G->A[v];
    while (e != (Edge *) 0){
      e1 = e->mate;
      if (Dist[e1->t] == -1 && e1->f < e1->c){
	Dist[e1->t] = Dist[v] + 1;
	S[++h] = e1->t;
      }
      e = e->next;
    }
  }

  if (Dist[v2] == -1){
    h = t = 0;
    S[0] = v2;
    Dist[v2] = G->size;
    while (t <= h){
      v = S[t++];
      e = G->A[v];
      while (e != (Edge *) 0){
	e1 = e->mate;
	if (Dist[e1->t] == -1 && e1->f < e1->c){
	  Dist[e1->t] = Dist[v] + 1;
	  S[++h] = e1->t;
	}
	e = e->next;
      }
    }
  }

}
