#include <stdio.h>
#include <math.h>

/* This program generates integer points within a circle centered at */
/* origin with radius RADIUS.  N is number of points, and alpha and  */
/* delta are parameters. RNG seed is input. Roughly, alpha is probablity  */
/* of staying in cluster at each step, and delta is cluster diameter.     */
/* Examples:                                                              */
/*   alpha, delta = (1,1) = (0,0): uniform distribution (for large N).    */
/*   alpha, delta = (.9,.01): about N/10  tiny clusters                   */ 
/*   alpha, delta = (.99,.1): about N/100 visible clusters                */ 

/* By Catherine McGeoch 1/91  netflow@dimacs.rutgers.edu   */  
/*                            ccm@cs.amherst.edu           */

#define RADIUS 100000000
#define     PI 3.141592653589793238462643383279
#define HALFPI 1.570796326794896619231321691639

double erand48();           /* RNG returns double-precision reals       */
                            /* from [0.0, 1.0).  Seed is passed as      */
                            /* a parameter, allowing independent        */
                            /* streams of numbers.  A simpler source    */  
                            /* of random numbers will do, if necessary. */ 

unsigned short xstream[3];  /* Two random number streams  */ 
unsigned short ystream[3];

double topprob;
double alpha, delta;  /* Parameters for the step distribution.     */
                      /*                 _______                   */
                      /*                 |     |                   */
                      /*                 |  a  |                   */
                      /* ________________|_____|__________________ */
                      /* |_______________|__b__|_________________| */

                      /* alpha is area (a+ b). Input.              */
                      /* delta is proportion                       */
                      /*   (high box width)/(total width). Input.  */
                      /* topprob is area a. Calculated in main.    */


/*-------------------------GenDistance-------------------------*/ 
double GenDistance(len, pos) 

/* Generates a random value on [-len/2, len/2] from a step     */
/* distribution centered at pos. Uses parameters alpha, delta, */ 
/* topprob described above. len is the length of the chord.    */

double pos, len;
{
   double z;
   double halflen; 

   /* Generate point on [-len/2, len/2].  With probability topprob (from   */
   /* main) scale to [-delta*len/2, delta*len/2]. Then translate and wrap.  */
   
   z = (erand48(xstream)-0.5) * len; 
   if (erand48(ystream) <= topprob) z = z*delta;
   z = z + pos;                        /* Translate by pos     */ 
   halflen = len/ 2.0;                 /* Wrap if z goes past end */ 
   if (z < -halflen)  z = z + len;
   else if (z > halflen) z = z - len;
/* printf("%lf  %lf   %lf \n", len, pos, z); */ 

   return z; 
} 

main (argc,argv)
int argc;
char *argv;
{

  double x,y;        /* current point */ 
  double r,s;        /* current point in new coordinate system */ 
  double rcos, rsin; /* cos and sin of rotate angle */ 
  double sum, t1, t2; /* temps */ 
  double len,z ;     /* chord length and point position on chord */ 
  double GenDistance(); 

  void srand48();    /* Used to Initialize the two RNG streams  */ 
  long lrand48();    /* Returns long integer in 0..2^31 - 1*/  
  long seed; 

  register int i;
  int numpts; 
   
  /* Get input parameters */ 
    fprintf(stderr, "Enter N, alpha, delta, seed\n");
    scanf("%d %lf %lf  %d", &numpts, &alpha, &delta, &seed);

   /*Calculate topprob from alpha, delta.*/
   topprob = alpha -(delta*(1.0 - alpha))/(1.0 - delta); 

  /* Initialize two random number streams */   
     srand48(seed); 
     for (i=0; i< 3 ; i++) {
          xstream[i] = lrand48() >> 16; 
	  ystream[i] = lrand48() >> 16; 
	} 

  /* Generate Starter Point in unit circle centered at (0,0) */
    do {
        x = erand48(xstream)*2.0-1.0; 
        y = erand48(ystream)*2.0-1.0; 
      } while ((x*x + y*y) > 1.0); 

   /* Generate Points */
     for (i=0; i< numpts; i++) {

        /* Current point has coordinates x,y.  Generate random direction  */
        /* and then a point on the chord through x,y using alpha, delta   */ 

        /* Convert to r,s coordinate system. We want r axis perpendicular  */
        /* to the chord passing through current point at angle theta, for  */
        /* random theta from [0, pi]. So, rotate axes by random angle     */ 
        /* on [-pi/2, pi/2].  We just need sin and cos of the rotate angle.*/ 

        /* Generate rotate angle on [-pi/2, pi/2] by finding a random      */
        /* point in unit semicircle.  Get sin and cos from rise and run.   */ 
        do {
          t1 = erand48(xstream);
          t2 = erand48(ystream)*2.0 - 1.0;
          sum = t1*t1 + t2*t2;        
        }  while (sum > 1.0);
        sum = sqrt(sum);
        rcos = t1/sum;
        rsin = t2/sum;

        /* Find coordinates of current point in the r,s system */ 
        r = x*rcos - y*rsin;
        s = x*rsin + y*rcos;

        /* Chord is perpendicular to r-axis.  Generate a new point on chord */ 
        len = 2.0*sqrt(1.0- r*r); 
        z = GenDistance(len, s); 
      
        /* New point is r,z.  Find its coordinates in the x,y system.   */ 
        /* Note cos(rotate) = cos(-rotate), -sin(rotate) = sin(-rotate) */
        rsin = -rsin;
        x = r*rcos - z*rsin;
        y = r*rsin + z*rcos;

/*        printf("%lf    %lf\n", x,y);  */
        printf("%d  %d \n", (int) (x*RADIUS), (int) (y*RADIUS)); 

    }/* for each point */ 

} /* main*/ 
