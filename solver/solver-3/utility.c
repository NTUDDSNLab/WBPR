/* utility.c */

#include "graph.h"

char *Alloc(n)
int n;
{
  char *malloc(), *p;

  if ((p = malloc(n)) == 0)
    Barf("Out of space");

  return p;
}

Barf(s)
char *s;
{
  fprintf(stderr, "%s\n", s);
  exit(-1);
}

int EOF_Test(f)
FILE *f;
{
  char c, ReadChar();

  c = ReadChar(f);
  if (c == EOF)
    return TRUE;
  ungetc(c, f);
  return FALSE;
}


int SkipLine(f)
FILE *f;
{
  char c;

  do
    c = getc(f);
  while (c != EOF && c != '\n');

}

/* Skip whitespace */
Skip(f)
FILE *f;
{
  char c;
 
  while (isspace(c = getc(f)))
    ;
  ungetc(c,f);
}

   
/* Get a string terminated by whitespace */
int GetString(f, buff)
FILE *f;
char *buff;
{
  char c;

  Skip(f);
  while (!isspace(c = getc(f)))
    *buff++ = c;
  *buff = 0;
}

int Strcmp(s1, s2)
char *s1, *s2;
{
  while (*s1 && *s2){
    if (*s1++ != *s2++)
      return FALSE;
  }
  return *s1 == *s2;

}



StrAppend(s1, s2, s3)
char *s1, *s2, *s3;
{
  while (*s1)
    *s3++ = *s1++;
  while (*s2)
    *s3++ = *s2++;
  *s3 = 0;
}


int GetInt(f)
FILE *f;
{
  char c, ReadChar();
  int v, sign;

  c = ReadChar(f);
  sign = FALSE;
  v = 0;

  if (c == '-'){
    sign = TRUE;
    c = getc(f);
  }
  while (isdigit(c)){
    v = 10*v + (c - '0');
    c = getc(f);
  }
  if (sign)
    v = -1*v;

  ungetc(c, f);
  return v;
}

PutInt(i, f)
int i;
FILE *f;
{
  char c;
  int d;

  if (i == 0){
    putc('0', f);
    return;
  }

  if (i < 0){
    putc('-', f);
    i = -1*i;
  }
  
  d = 1;
  while (d <= i)
    d *= 10;
  d /= 10;
  while (d > 0){
    c = i / d;
    i %= d;
    d /= 10;
    putc('0'+c, f);
  }
  
}

char ReadChar(f)
FILE *f;
{
  char c;

  do {
    c = getc(f);
  } while (isspace(c));
  return c;
}

int Min(x, y)
int x, y;
{
  return (x > y) ? y : x;
}

int Max(x, y)
int x, y;
{
  return (x > y) ? x : y;
}

int Abs(x)
int x;
{
  return (x > 0) ? x : -x;
}

/* Open a file for reading - if the file doesn't exist,
then the extention .max is tried, if that doesn't work
then the data directory is checked - with or without the extention.
*/

FILE *OpenFile(c)
char *c;
{
  FILE *f;
  char buff1[100], buff2[100];

  if ((f = fopen(c,"r")) != NULL)
    return f;

  StrAppend(c,".max",buff1);
  if ((f = fopen(buff1,"r")) != NULL)
    return f;

  StrAppend(DATA_DIRECTORY,c,buff1);
  if ((f = fopen(buff1,"r")) != NULL)
    return f;

   StrAppend(buff1,".max",buff2);
   return fopen(buff2,"r");

  

}


Queue *MakeQueue(n)
int n;
{
  Queue *Q;

  Q = (Queue *) Alloc(sizeof(Queue));

  Q->data = (int *) Alloc(n * sizeof(int));
  Q->tail = 0;
  Q->head = 0;

  Q->size = n;

  return Q;
}

int Dequeue(Q)
Queue *Q;
{
  int v;

  if (Q->tail == Q->head)
    Barf("Attempt to dequeue from empty queue");

  v = Q->data[Q->head];
  Q->head = (Q->head == Q->size - 1) ? 0 : Q->head + 1;

  return v;
}

Enqueue(Q, k)
Queue *Q;
int k;
{
  if (Q->head == Q->tail + 1 ||
      (Q->tail == Q->size - 1 && Q->head == 0))
    Barf("Queue overfull");

  Q->data[Q->tail] = k;
  Q->tail = (Q->tail == Q->size - 1) ? 0 : Q->tail + 1;
}


int QLast(Q)
Queue *Q;
{
  return (Q->tail > 0) ? Q->data[Q->tail - 1] : Q->data[Q->size - 1];
}

int QSize(Q)
Queue *Q;
{
  return (Q->tail >= Q->head) ? Q->tail - Q->head 
                              : Q->tail - Q->head + Q->size;
}

QueueEmpty(Q)
Queue *Q;
{
  return Q->head == Q->tail;
}

PrintQueue(Q)
Queue *Q;
{
  int i;

/*  printf("QueueSize: %d; ", QSize(Q)); */

  if (Q->head == Q->tail)
    printf("Empty Queue");
  else if (Q->head > Q->tail){
    for (i = Q->head; i < Q->size; i++)
      printf("%3d ", Q->data[i] + 1);
    for (i = 0; i < Q->tail; i++)
      printf("%3d ", Q->data[i] + 1);
  }
  else {
    for (i = Q->head; i < Q->tail; i++)
      printf("%3d ", Q->data[i] + 1);
  }
  printf("\n------------------------------\n");
}


Stack *MakeStack(n)
int n;
{
  Stack *S;

  S = (Stack *) Alloc(sizeof(Stack));

  S->data = (int *) Alloc(n * sizeof(int));
  S->ptr = 0;

  S->size = n;

  return S;
}

StackPush(S, v)
Stack *S;
int v;
{
  if (S->ptr >= S->size)
    Barf("Stack overflow");

  S->data[S->ptr++] = v;

}

int StackPop(S)
Stack *S;
{
  if (S->ptr == 0)
    Barf("Pop from empty stack");

  return S->data[--S->ptr];
}

int StackEmpty(S)
Stack *S;
{
  return S->ptr == 0;
}

int StackSize(S)
Stack *S;
{
  return S->ptr;
}

SortEdgeLists(G)
Graph *G;
{
  int i;
  Edge *e1, *l, *e, *ListInsert();

  for (i = 0; i < G->size; i++){
    l = (Edge *) 0;
    e = G->A[i];
    while (e != (Edge *) 0){
      e1 = e->next;
      l = ListInsert(e, l);
      e = e1;
    }
    G->A[i] = l;
  }
}


Edge *ListInsert(e, l)
Edge *e, *l;
{
  Edge *l1;

  e->next = (Edge *) 0;
  if (l == (Edge *) 0)
    return e;
  if (e->h <= l->h){
    e->next = l;
    return e;
  }
  l1 = l;
  while ((l1->next != (Edge *) 0) && (l1->next->h <= e->h))
    l1 = l1->next;
  e->next = l1->next;
  l1->next = e;
  return l;
}



RandomEdgeLists(G)
Graph *G;
{
  int i, c;
  Edge *e1, *l, *e, *ListPut();

  for (i = 0; i < G->size; i++){
    l = (Edge *) 0;
    e = G->A[i];
    c = 0;
    while (e != (Edge *) 0){
      e1 = e->next;
      l = ListPut(e, l, RandomInteger(0, c));
      e = e1;
      c++;
    }
    G->A[i] = l;
  }
}

Edge *ListPut(e, l, x)  /* Put e after element x of l */
Edge *e, *l;
int x;
{
  Edge *l1;
  int i;
  
  if (x == 0){
    e->next = l;
    return e;
  }

  l1 = l;  
  for (i = 0; i < x - 1; i++)
    l1 = l1->next;
  e->next = l1->next;
  l1->next = e;
  return l;
}
