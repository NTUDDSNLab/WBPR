#include<stdio.h>
#include<math.h>

/*-----------------------------------------------------------------*/
/*  Generates instances for maximum flow according to DIMACS       */ 
/*  Challenge format.  Instances have random edges with uniform    */
/*  capacities.  Will be very slow for near-complete networks.     */
/*  It is not claimed that these networks are interesting          */ 
/*  C. McGeoch 10/15/90  */

/*  You may need to insert your own random number generators.     */
/*     double drand48();  returns doubles from (0.0, 1.0]         */
/*     void srand48(seed);  initializes RNG with seed             */
/*     long seed;                                                 */

/*  Example input:         */
/*  nodes 1000             */
/*  arcs  2500             */
/*  seed  818182717        */

/*  Seed command is optional.                           */
/*  nodes  N  : specifies N nodes                       */ 
/*  arcs   A  : specifies A arcs                        */
/*  seed   S  : RNG seed; default system timer          */ 
/*------------------------------------------------------*/  

#define Assert( cond , msg ) if ( ! (cond) ) {printf("msg\n"); exit(); } ; 
#define MAXNODES 1000
#define MAXCAP  100000

#define TRUE 1
#define FALSE 0

typedef char string[50];

/* BST for Set searches */ 
typedef struct node_type {
  int val;
  struct node_type *left;
  struct node_type *right;
} treenode; 

int arc_size; 
treenode *root; 

/* Random Number Functions */ 
double drand48();
void srand48(); 

/* Global Parameters */
long seed;
int nodes, arcs;
int rand_seed;        /*boolean*/ 
int seed; 
int minarcs, maxarcs; 

string cmdtable[10];
int cmdtable_size; 
int nodelist[MAXNODES]; 
int listsize; 

/*----------------- Initialize tables and data  */ 
void init()
{ 
  int i; 
  
  cmdtable_size = 3;
  strcpy(cmdtable[0], "sentinel.spot");
  strcpy(cmdtable[1], "nodes" );   /* required */ 
  strcpy(cmdtable[2], "arcs"  );
  strcpy(cmdtable[3], "seed");
  
  nodes = MAXNODES;
  arcs = MAXNODES;  
  rand_seed = TRUE;

  root = NULL; 
  arc_size = 0; 

}   

/* ---------------- Random number utilities */

/* Initialize rng */  
void init_rand(seed) 
int seed; 
{
  srand48(seed);
    }

/* Return an integer from [1..max] */  
int rand_int(max) 
int max;
{
  double x; 
  int i; 
  
  x = drand48();
  i = (double) x * max + 1.0;
  return(i);
}

/*--------------------------bst for selection wo replacement */ 

int inset(arc, ptr)
int arc;
{
  treenode *current;

  current = root; 
  while(current != NULL) {
    if (current->val > arc) { 
      current = current->left;
    }
    else if (current->val < arc) {
      current = current->right;
    }
    else 
      return(TRUE);
  }/*while*/

  return(FALSE);

}/*inset*/

/* insert arc into tree--assumes its not already in */ 

void insert(arc)
int arc;
{
  treenode *current;
  treenode *parent;

  if ( arc_size ==  0) {  /* first insertion */
    root = (treenode *) malloc(1*sizeof(treenode));
    root->val = arc;
    root->left = NULL;
    root->right= NULL; 
  }else{
    current = root;

    while (current != NULL) {
      parent = current; 
      /* search for the empty spot, saving parent */
      if (arc < parent->val) { 
	current = parent->left;
	}
      else if (arc > parent->val) {
	current = parent->right;
      }

    }/* while */ 
  
    current = (treenode *) malloc(1*sizeof(treenode));
    current->val = arc;
    current->left = NULL;
    current->right = NULL; 

    if (arc < (parent->val)) parent->left = current;
    else parent->right = current;
  }/*else not new  */

    arc_size++; 
} /* insert */

/* Returns random arcs without replacement */  
/* arc is represented by a number from 1..(n^2-n) */
/* (since self-arcs are not allowed) */

next_arc() 
  {
   int i; 
   int out; 
   
   do {
   i = rand_int(maxarcs);
   } while ( inset(i) == TRUE );
   insert(i); 
   return(i);
}
   
/*------------------Command input routines  */ 
int lookup(cmd)
{
 int i;
 int stop;

 strcpy( cmdtable[0], cmd);  /* sentinel */ 
 stop = 1;
 for (i = cmdtable_size; stop != 0; i--) stop = strcmp(cmdtable[i], cmd);

 return ( i + 1 ); 
}/*lookup*/

/*---------------------- Read command lines */ 
void get_input()
{
char cmd[50], buf[50];
int index;
int i; 

  while (scanf("%s", cmd ) != EOF) {
    fgets(buf, sizeof(buf), stdin);
    index = lookup(cmd);
    switch(index) {

    case 0:  { fprintf(stderr, "%s: Unknown command. Ignored.\n", cmd);
	       break;
	     }
    case 1:  {sscanf( buf , "%d", &nodes); 
	      Assert( 0<=nodes && nodes<=MAXNODES , Nodes out of range. );
	      maxarcs =  (nodes*nodes) - nodes; 
	      break;
	    }
    case  2: {sscanf (buf , "%d", &arcs);
	      break;
	    }
    case 3: { sscanf( buf, "%d", &seed);
	       rand_seed  = FALSE;
	       break;
	    }

    }/*switch*/
  }/*while*/

  /* Do sanity checks */ 
  Assert( 0 <= arcs && arcs <= maxarcs , Bad arcs value. ); 

}/*input*/
/* ---------------------------Report parameters  */

void report_params()
{
  printf("c nodes %d\n", nodes);
  printf("c arcs %d \n", arcs);
  if (rand_seed == TRUE) printf("c random seed\n");
  else printf("c seed %d\n", seed);
}

/*--------------------------- Generate and print out network  */ 
void generate_net()
{
 int n, a, i;
 int cap; 
 int src, dst; 
 int t, s; 

  if (rand_seed == TRUE) init_rand((int) time(0));
  else init_rand(seed); 

  /* Print first line and some comments  */

  printf("p max  \t %d \t %d \n", nodes, arcs);
  printf("c Maximum flow problem with %d  nodes and %d arcs\n", 
          nodes, arcs);
  printf("c Randomly generated---very little structure \n");
  printf("c Max arc capacity %d\n", MAXCAP);

  report_params();

 /* Generate distinct source and sink */
 
  s = rand_int(nodes);
  printf("n \t %d \t s\n", s);

  t = rand_int(nodes-1);
  if (s == t) t = nodes; 
  printf("n \t %d \t t\n", t);
 
  /* Generate arcs: multi-arcs and self-arcs not allowed */ 
  for (i = 0; i < arcs; i++) { 
   a  = next_arc();             /* get an unused arc inde: */
                                /* from 1..(nodes*nodes-nodes) */ 
   
   src = ((a-1) / (nodes-1)) + 1;  /* row number in 1..n */
   dst = (a % (nodes-2)) + 1;      /* col number in 1..n-1 */
   if (src ==  dst) {         /* move a self-arc [i,i] to [i,n] */ 
      dst = nodes;
    }

   cap = rand_int(MAXCAP); 

   printf("a \t %d \t %d \t %d\n", src, dst, cap); 
 }/* for each arc */

}/*generate_net*/

main()
{

    init(); 

    get_input();

    generate_net(); 

  } 
