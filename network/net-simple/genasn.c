#include<stdio.h>
#include<math.h>

/*-----------------------------------------------------------*/
/*  Generates instances for assignment according to DIMACS   */ 
/*  Challenge format.  Each instance is a complete           */
/*  bipartite graph with random edge costs.                  */
/*  It is not claimed that the instances are interesting.    */ 
/*  C. McGeoch 10/15/90  */

/*  You may need to insert your own random number generators. */
/*     double drand48();  returns doubles from (0.0, 1.0]   */
/*     void srand48(seed); initializes RNG with seed        */  
/*     long seed;                                           */

/*  Example input:                 */
/*  nodes  1000                    */
/*  sources  491                   */
/*  seed   828272727               */

/*  All commands are optional; have defaults; can appear in any  */
/*  order.                                                       */ 
/*    nodes   N   : specifies N nodes; default MAXNODES          */
/*    sources S   : specifies S sources; default 1               */ 
/*    seed    X   : specifies X a random seed; default use timer */ 
/*---------------------------------------------------------------*/

#define Assert( cond , msg ) if ( ! (cond) ) {printf("msg\n"); exit(); } ; 
#define MAXNODES 10000
#define MAXCOST  100000

#define TRUE 1
#define FALSE 0

typedef char string[50];

/*  RNG Declarations  */
double drand48();
void srand48(); 
long seed;

/* Global Parameters */
int nodes, arcs;      
int sources, sinks;   /* number of nodes each type */ 
int rand_seed;        /*boolean flag*/ 
int minarcs, maxarcs; /*bounds determined by nodes */ 

/* Stuff for reading input commands */
string cmdtable[10];
int cmdtable_size; 

/* Generating random nodes without replacement */
int nodelist[MAXNODES]; 
int listsize; 

/*--------------- Initialize tables and data  */ 
void init()
{ 
  int i; 
  
  cmdtable_size = 3;
  strcpy(cmdtable[0], "sentinel.spot");
  strcpy(cmdtable[1], "nodes" );   /* required */ 
  strcpy(cmdtable[2], "seed");
  strcpy(cmdtable[3], "sources"); 
  
  nodes = MAXNODES;
  rand_seed = TRUE;
  sources = 1;
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
   nodelist[listsize] = out; 
   listsize--;

   /* nodes out are saved in nodelist[listsize+1 .. nodes] */ 
   return(out); 
} 
   
/*------------------Command input routines  */ 

/* Lookup command in table */
int lookup(cmd)
{
 int i;
 int stop;
 strcpy( cmdtable[0], cmd);  /* sentinel */ 
 stop = 1;
 for (i = cmdtable_size; stop != 0; i--) stop = strcmp(cmdtable[i], cmd);
 return ( i + 1 ); 
}/*lookup*/


/* Get and process input commands  */ 
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
	      Assert( 1<=nodes && nodes<=MAXNODES , Nodes out of range. );
	      break;
	    }
    case 2: { sscanf( buf, "%d", &seed);
	       rand_seed  = FALSE;
	       break;
	    }
    case 3: { sscanf( buf, "%d", &sources);
	      Assert( 1<=sources && sources <= nodes, Sources out of range. );
	      break; 
	    }
    }/*switch*/
  }/*while*/

sinks = nodes - sources;
arcs = sinks * sources; 

/* Initialize nodelist for this problem size */
for (i = 1; i <= nodes; i++)  nodelist[i] = i;
listsize = nodes; 

}/*get_input*/

/*---------------------------Report parameters  */

void report_params()
{
  printf("c Assignment flow problem\n"); 
  printf("c Max arc cost %d\n", MAXCOST);
  printf("c nodes %d\n", nodes);
  printf("c sources %d \n", sources);
  if (rand_seed == TRUE) printf("c random seed\n");
  else printf("c seed %d\n", seed);
}

/*--------------------------- Generate and print out network  */ 
void generate_net()
{
 int n, x, j, i;
 int arcsleft; 
 int low, cap, cost; 
 int src, dst; 
 int arccount; 

  if (rand_seed == TRUE) init_rand((int) time(0));
  else init_rand(seed); 

  /* Print first line and report parameters  */
  printf("p asn  \t %d \t %d \n", nodes, arcs);
  report_params();

  /* Generate  source nodes */
  for (i=0; i <  sources ; i++) {
    n = next_node();
    printf("n \t %d\n", n);
  }

 /* Generate complete bipartite graph   */  
 /* sources in nodelist[listsize+1..n] */
 /* sinks in nodelist[1..listsize] */ 

 for (i=listsize+1; i <= nodes; i++){
   src = nodelist[i];
   for (j = 1; j <= listsize; j++) {
     dst = nodelist[j]; 
     cost = rand_int(MAXCOST);
     printf("a \t %d \t %d \t %d\n", src, dst, cost); 
   }
 }/* for each source */ 

}/*generate_net*/

/*--------------------main */ 
main()
{

    init(); 

    get_input();

    generate_net(); 

  } 
