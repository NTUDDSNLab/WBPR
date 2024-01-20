#include <stdio.h>
#include <math.h>

/*  This program produces fractal point sets defined by affine   */
/*  transformations; see M. Barnsley and A. Sloan, ``A better    */
/*  way to compress images,'' Byte, January 1988, pp 215-223.    */
/*  A command file is read from standard input.  An example      */
/*  command file is given below.                                 */
/*    c       This is a comment line, for humans  */ 
/*    points  500                                 */
/*    seed    8338382                             */
/*    trans   .05    0     0    0   .5  0  .5     */
/*    trans   .15   .1     0    0   .1  0  .2     */
/*    trans   .4    .42  -.42 .42  .42  0  .2     */ 
/*    trans   .4    .42   .42 -.42 .42  0  .2     */

/*  points  Gives number of points to generate. If absent, default is 100  */
/*  seed    Gives random number seed. If absent, machine time is seed      */
/*  trans   Reading data as ``P  a b c d e f'', it means:                  */ 
/*             With probability P apply the transformation                 */
/*                 |a  b| * |x|  + |e|                                     */
/*                 |c  d|   |y|    |f|                                     */
/*             to the current point (x,y).  That is, multiply the vector   */
/*             [x,y] by the 2x2 matrix [a,b/c,d], then add vector [e,f].   */
/*          The probabilities must sum to 1. A maximum of 8 transformations*/
/*          can appear in the input.  The example above will produce the   */
/*          well-known ``fractal broccoli.''                               */  
/*  Written by C. C. McGeoch, DIMACS, June 1991                            */ 
/*  Output format corresponds to DIMACS Challenge Specifications           */

#define SCALE 1000000      /* scale points to integer range  */ 
#define MAXTRANS 8         /* max number of affine transformations */

typedef char string[80]; 

typedef struct {          /* holds specs of a transformation */ 
  double prob;
  double a, b, c, d, e, f;
} trans_type;

trans_type transform[MAXTRANS];  /* holds the transforms from input */ 
tr_size;                         /* number of transforms in input */ 

int points;       /* number of points generated */ 

string cmdtable[10];  /* input command tables */
int cmdtable_size; 

long seedval;     /* random number generator seed */ 
double drand48(); /* system rng for reals in (0, 1]*/ 
void srand48();   /* initialize rng */

/*----------Substitute your own system random generators here---------*/ 
void initrand(seed)
long seed; 
{
  srand48(seed);
} 

double myrand()
{
   return(drand48());
} 

/*--------- Input Handlers----------------------*/

/*---------tabinit------------------------------*/
/* initialize command interpreter and parameters*/

void tabinit()
{
  cmdtable_size = 4;
  strcpy(cmdtable[0], "sentinel.spot");
  strcpy(cmdtable[1], "points");
  strcpy(cmdtable[2], "seed");
  strcpy(cmdtable[3], "trans");
  strcpy(cmdtable[4], "c"); 
 }

/*--------transinput---------------------------*/
/* get input and set parameters                */ 

void transinput() {
  char cmd[20], buf[80];
  int index;
  int i, j;
  int stop;
  double probability, cumprob; 
  string comment; 
 
  /* default values */ 
  points = 100; 
  seedval = (long) time(0);
  tr_size = 0; 
  cumprob = 0.0;   /* cumulative probability */ 
 
  /* get input lines, find first string */
  while (scanf("%s", cmd) != EOF ) {
    fgets(buf, sizeof(buf), stdin);  /*read first string in line */ 
    
    /* look up command in command table */
    strcpy(cmdtable[0], cmd); /* insert sentinel */ 
    stop = 1;
    for (i=cmdtable_size; stop != 0; i--) stop = strcmp(cmdtable[i], cmd);
    index = i+1; 
    switch(index) {
    case 0: { printf("%s: Unknown command. Ignored\n", cmd);
	      break;
	    }
    case 1: { sscanf( buf, "%d", &points); 
	      break;
	    }
    case 2: { sscanf( buf, "&d", &seedval);
	      break;
	    }
    case 3: { sscanf( buf, "%lf %lf %lf %lf %lf %lf %lf", 
		     &probability, 
		     &transform[tr_size].a,
		     &transform[tr_size].b,
		     &transform[tr_size].c, 
		     &transform[tr_size].d,
		     &transform[tr_size].e,
		     &transform[tr_size].f );
              cumprob = probability + cumprob; 
	      transform[tr_size].prob = cumprob; 
	      tr_size++;
	      break;
	    }
    case 4: { sscanf( buf, "%s", comment);
	      break; 
	    }
    }/*switch*/ 
  }/* while scanf */ 
}/*transinput */ 


/*---------main--------------------------------*/
/*  Generate the points                        */

void main ()
{
  double x, y;
  double xnew,ynew;
  double pr; 
  int xint , yint;
  int i, j; 

  tabinit();
  transinput();
  /* DIMACS Challenge Headers */
  printf("p geom %d  2\n", points); 
  printf("c  A fractal-type picture produced by fractal.c \n");

  x = 0.0; 
  y = 0.0;
  /* ignore the first 10 points  */ 
  for (i = 0; i< 10; i++) {
      pr = myrand();
      for (j = 0; j < tr_size; j++) {
	if  (pr <= transform[j].prob) break;
      }
      xnew = transform[j].a * x + transform[j].b * y + transform[j].e;
      ynew = transform[j].c * x + transform[j].d * y + transform[j].f;
      x = xnew;
      y = ynew;
    } 

  /* now generate points */ 
  for (i = 0; i< points; i++) {
      pr = myrand();
      for (j = 0; j < tr_size; j++) {
	if  (pr <= transform[j].prob) break;
      }
      xnew = transform[j].a * x + transform[j].b * y + transform[j].e;
      ynew = transform[j].c * x + transform[j].d * y + transform[j].f;
      x = xnew;
      y = ynew;
      xint = (int) (SCALE* x); 
      yint = (int) (SCALE* y); 
      printf("v  %d  %d\n", xint , yint); 
    } 
}/*main*/ 
  

