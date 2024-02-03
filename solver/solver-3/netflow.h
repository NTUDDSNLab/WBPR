 /* netflow.h */
  int freq;


#define DFS          1
#define BFS          2
#define MAX_GAIN     3
#define DINIC        4
#define DINIC_NEW    5
#define KARZANOV     6
#define GOLDBERG_1   7
#define GOLDBERG_2   8
#define GOLDBERG_3   9
#define GOLDBERG_4  10
#define GOLDBERG_5  11

#define INFINITY 10000000

int Pred[MAX_N];

struct {
  int D[MAX_N], hptr[MAX_N], a[MAX_N];
  int size, range;
} Heap;
  

typedef struct lnode {
  struct lnode *next;
  int v;
} LNode;

typedef struct {
  LNode *data[2*MAX_N], *freelist;
  int max, size, elts;
} IHeap;
  
 


