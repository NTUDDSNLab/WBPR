/* ghead.h */

#define DATA_DIRECTORY "/local/users/anderson/netflow/data/"

#define FAILURE    0
#define SUCCESS    1
#define FALSE      0
#define TRUE       1


#define MAX_N     20000  
/* #define MAX_N 60 */

#define MAX_CAP   100000000

/* Dimacs problem types */
#define UNDEFINED        0
#define MINCOSTFLOW      1
#define MAXFLOW          2
#define ASSIGNMENT       3

typedef struct enode {
  struct enode *next;
  struct enode *mate;
  int c;
  int f;
  int h;
  int t;
  int flag;
} Edge;


typedef struct {
  Edge *A[MAX_N];
  int V[MAX_N];
  int size;
  int max_v;
  int edge_count;
} Graph;

typedef struct {
  int head, tail, size;
  int *data;
} Queue;

typedef struct {
  int ptr, size;
  int *data;
} Stack;




