#include<stdio.h>
#include<math.h>

/*---------------------------------------------------------------------*/
/*  Generates instances for minimum cost flow according to DIMACS      */ 
/*  Challenge format.  Each node has indegree at least 1 and outdegree */ 
/*  at least 1. It is not claimed that these instances are interesting.*/ 
/*  Costs, capacities, supplies, and demands are uniform integers.     */ 
/*  Multiple arcs may occur.                                           */  

/*  C. McGeoch 10/15/90  */

/*  You may need to insert your own random number generators.    */
/*      double drand48(); returns doubles from (0.0, 1.0]        */
/*      void srand48(seed); initalizes RNG with seed             */
/*      long seed;                                               */

/*  Example input:         */
/*  nodes 100              */
/*  arcs  500              */
/*  supply 10              */
/*  demand 20              */
/*  seed  189038           */
/*  (control-d)            */

/*  First command required; others are optional   */ 

/*  nodes N:  specifies N nodes                                   */ 
/*  arcs  A:  specifies A arcs; between 2N and N(N-1); default 2N */ 
/*  supply S: number of supply nodes; default 1                   */
/*  demand D: number of demand nodes; default 1                   */
/*  seed X:   RNG seed; default system timer                      */ 
/*----------------------------------------------------------------*/ 

#define Assert( cond , msg ) if ( ! (cond) ) {printf("msg\n"); exit(); } ; 
#define MAXNODES 1000
#define NODEMAX 100000   /* max supply/demand per node */ 
#define MAXCOST 100000
#define MAXCAP  100000

#define TRUE 1
#define FALSE 0

typedef char string[50];

double drand48();
void srand48(); 

/* Global Parameters */
long seed;
int nodes, arcs;
int supply, demand;   /* number of nodes each type */ 
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
  
  cmdtable_size = 5;
  strcpy(cmdtable[0], "sentinel.spot");
  strcpy(cmdtable[1], "nodes" );   /* required */ 
  strcpy(cmdtable[2], "arcs"  );
  strcpy(cmdtable[3], "seed");
  strcpy(cmdtable[4], "supply"); 
  strcpy(cmdtable[5], "demand"); 
  
  nodes = MAXNODES;
  arcs = MAXNODES*2;  
  rand_seed = TRUE;
  supply = 1;
  demand = 1; 

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

/* Returns random nodes without replacement */  
/* nodelist is initialized in getinput  */ 

int next_node() 
  {
   int i; 
   int out; 

   i = rand_int(listsize);
   out = nodelist[i];
   nodelist[i] = nodelist[listsize];
   listsize--;

   return(out); 

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
    case 0:  { printf("%s: Unknown command. Ignored.\n", cmd);
	       break;
	     }
    case 1:  {sscanf( buf , "%d", &nodes); 
              Assert( 2 < nodes, Nodes must be at least 3.); 
	      Assert( nodes<=MAXNODES ,  Too many nodes. );
	      maxarcs =  (nodes*nodes) - nodes; 
	      minarcs = 2*nodes; 
              if (minarcs > maxarcs) minarcs = maxarcs; 
              arcs = minarcs; 
	      break;
	    }
    case  2: {sscanf (buf , "%d", &arcs);
	      break;
	    }
    case 3: { sscanf( buf, "%d", &seed);
	       rand_seed  = FALSE;
	       break;
	    }
    case 4: { sscanf( buf, "%d", &supply);
	      Assert( 1 <= supply && supply <= nodes, Supply out of range. );
	      break; 
	    }
    case 5: { sscanf( buf, "%d", &demand);
	      Assert( 1 <= demand && demand <= nodes, Demand out of range. );
	      break; 
	    }
    }/*switch*/
  }/*while*/

  /* Do sanity checks */ 
  Assert( minarcs <= arcs && arcs <= maxarcs , Bad arcs value. ); 
  Assert( demand + supply <= nodes,  Too many supply + demand nodes. );

  /* Initialize for problem instance */
  for (i = 1; i <= nodes; i++)  nodelist[i] = i;
  listsize = nodes; 

}/*input*/
/* ---------------------------Report parameters  */

void report_params()
{
  printf("c nodes %d\n", nodes);
  printf("c arcs %d \n", arcs);
  printf("c supply %d \n", supply);
  printf("c demand %d \n", demand);
  if (rand_seed == TRUE) printf("c random seed\n");
  else printf("c seed %d\n", seed);
}

/*--------------------------- Generate and print out network  */ 
void generate_net()
{
 int n, x, i;
 int arcsleft; 
 int low, cap, cost; 
 int src, dst; 

  if (rand_seed == TRUE) init_rand((int) time(0));
  else init_rand(seed); 

  /* Print first line and some comments  */

  printf("p min  \t %d \t %d \n", nodes, arcs);
  printf("c Minimum-cost flow problem \n");  
  printf("c Automatically generated---very little structure \n");
  printf("c Max node supply/demand %d\n", NODEMAX);
  printf("c Max arc cost %d\n", MAXCOST);
  printf("c Max arc capacity %d\n", MAXCAP);

  report_params();

  /* Report demand and supply nodes */
  for (i=0; i <  demand ; i++) {
    n = next_node();
    x = -rand_int(NODEMAX); 
    printf("n \t %d \t %d\n", n, x);
  }

  for (i=0; i< supply; i++) { 
    n = next_node();
    x = rand_int(NODEMAX); 
    printf("n \t %d \t %d \n", n, x);
  }

  /* Generate arcs: self-arcs not allowed */ 

  for (i=1; i <= nodes; i++){
     src = i;
     do {
        dst = rand_int(nodes);
	}while( dst == src); 
     low = 0;
     cap = rand_int(MAXCAP);
     cost = rand_int(MAXCOST);

     printf("a \t %d \t %d \t %d \t %d \t %d \n", src, dst, low, cap, cost); 

     dst = i;
     do {
       src = rand_int(nodes);
     } while ( dst == src); 
     low =0; 
     cap = rand_int(MAXCAP);
     cost = rand_int(MAXCOST); 

     printf("a \t %d \t %d \t %d \t %d \t %d \n", src, dst, low, cap, cost); 
   }/* for first 2n arcs */

    /* Generate the other arcs */
    arcsleft = arcs  - minarcs; 
    for (i =0 ; i<arcsleft; i++) {
      
      src = rand_int(nodes);
      do {
	dst = rand_int(nodes);
       }while(src==dst);
       low = 0;
       cap = rand_int(MAXCAP);
       cost = rand_int(MAXCOST); 

     printf("a \t %d \t %d \t %d \t %d \t %d \n", src, dst, low, cap, cost); 
     }/* for other of arcs */

}/*generate_net*/


main()
{

    init(); 

    get_input();

    generate_net(); 

  } 
