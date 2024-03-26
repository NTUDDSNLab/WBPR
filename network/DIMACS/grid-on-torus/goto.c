/**********************************************
 *  GOTO (Grid On TOrus generator)            *
 *  Copyright 1991 by Andrew V. Goldberg      *
 *  Deprtment of Computer Science, Stanford U *
 *  Use permited subject to a proper          *
 *  acknowledgement.                          *
 **********************************************
 */

/*
***********************************************

 To compile under UNIX, do "cc -O goto.c -lm"

***********************************************
*/

#include <values.h>
#include <stdio.h>
#include <math.h>

/* for random number generator */

#define B         13415821
#define MOD       100000000
#define MOD1      10000
#define MAXYCOST  8 

/* global variables */

long X, Y, XDEG, YDEG, seed;
long MAXCAP, SMALLCAP, RETCOST, MAXCOST,MAXDEG;
long N, M, M1, EXTRA_N, EXTRA_M;
long *cost;
long *capacity;
double ALPHA;
long S, T, SUPPLY;

/* random number generator following Sedgewick's book */
long mult(p,q)
long p,q;

{
  long p1,p0,q1,q0;

  p1 = p / MOD1; p0 = p % MOD1;
  q1 = q / MOD1; q0 = q % MOD1;
  return((((p0 * q1 + p1 * q0) % MOD1) * MOD1 + p0 * q0) % MOD);
  
}

long random_int(a, b)
long a, b;
{

  long r;

  r = b - a;

  seed = (mult(seed, B)+1) % MOD;
  return(a + (long) (((double) seed * (double) r) / (double) MOD));

}

long random_bit()

{
  
  return(random_int(0,1)); 

}

long random_capacity(dist) 
long dist;
/* returns random capacity conditioned by dist */

{
double ans;

  ans = (double) random_int(1, MAXCAP);
  ans = ans / pow((double) ALPHA, (double) dist-1);
  return((long) ceil(ans));

}

long random_cost(dir)
char dir;
/* returnes random cost conditioned by dist and u */

{
long ans;

  if (dir == 0) /* x direction */
    ans = random_int(0, MAXCOST);
  else
    ans = random_int(0, MAXYCOST);
  return(ans);

}

long grid_to_id(x, y)
long x,y;
{

return((y * X) + x + 1);

}

long node_loc(x, y)
long x,y;
{

return((y * X) + x);

}

long arc_loc(x1, y1, x2, y2)
long x1, y1, x2, y2;

{
long dist;

  if (y1 == y2) {
    if (x1 < x2)
      dist = x2 - x1;
    else
      dist = X - (x1 - x2);
    return((XDEG + YDEG) * node_loc(x1, y1) + dist - 1);
  }
  else {
    if (y1 < y2)
      dist = y2 - y1;
    else
      dist = Y - (y1 - y2);
    return((XDEG + YDEG) * node_loc(x1, y1) + dist + XDEG - 1);
  }
}

build_example(X,Y,XDEG,YDEG)
long X,Y,XDEG,YDEG;

{
long x,y,z,i,u,c;

  /* generate */
  for (y = 0; y < Y; y++) {
    for (x = 0; x < X; x++) {
      /* x-direction edges */
      for (i = 1; i <= XDEG; i++){
        z = (x+i) % X;
        u = random_capacity(i);
        c = random_cost(0);
        cost[arc_loc(x, y, z, y)] = c;
        capacity[arc_loc(x, y, z, y)] = u;
        if ((x >= z) && (grid_to_id(x,y) != T))
          SUPPLY += capacity[arc_loc(x, y, z, y)];
        if (grid_to_id(z,y) == T)
          SUPPLY += capacity[arc_loc(x, y, z, y)];
      }
      /* y-direction edges */
      for (i = 1; i <= YDEG; i++){
        z = (y+i) % Y;
        u = random_int(1, MAXCAP);
        c = random_cost(1); 
        cost[arc_loc(x, y, x, z)] = c;
        capacity[arc_loc(x, y, x, z)] = u;
      }
    }
  }
  
  /* print nodes */
  printf("n %8d %10d\n", S, SUPPLY);
  printf("n %8d %10d\n", T, -SUPPLY);

  /* print arcs */
  for (y = 0; y < Y; y++)
    for (x = 0; x < X; x++) {
      /* x direction */
      for (i = 1; i <= XDEG; i++){
        z = (x+i) % X;
        if (z > x) {
          printf("a %8d %8d %10d %10d %10d \n", 
                  grid_to_id(x, y),
                  grid_to_id(z, y),
                  0, capacity[arc_loc(x, y, z, y)],
                  cost[arc_loc(x, y, z, y)]);
	}
        else {
          if (grid_to_id(x, y) != T)
            printf("a %8d %8d %10d %10d %10d \n", 
                    grid_to_id(x, y),
                    T,
                    0, capacity[arc_loc(x, y, z, y)],
                    cost[arc_loc(x, y, z, y)]);
          if (grid_to_id(z, y) != S)
            printf("a %8d %8d %10d %10d %10d \n", 
                    S,
                    grid_to_id(z, y),
                    0, capacity[arc_loc(x, y, z, y)],
                    cost[arc_loc(x, y, z, y)]);
         }
      }
      /* y direction */
      for (i = 1; i <= YDEG; i++){
        z = (y+i) % Y;
        printf("a %8d %8d %10d %10d %10d \n", 
                grid_to_id(x, y),
                grid_to_id(x, z),
                0, capacity[arc_loc(x, y, x, z)],
                cost[arc_loc(x, y, x, z)]);
      }
    }
  if (EXTRA_N > 0) {
    /* connect extra nodes */
    printf("a %8d %8d %10d %10d %10d \n",
            S, X*Y+1,
            0, SMALLCAP, MAXCOST/2);
    printf("a %8d %8d %10d %10d %10d \n",
            N, T,
            0, SMALLCAP, MAXCOST/2);
    for (i = 1; i < EXTRA_N; i++)
      printf("a %8d %8d %10d %10d %10d \n",
              X*Y+i, X*Y+i+1,
              0, SMALLCAP, MAXCOST/2);

    for (i = 1; i<= EXTRA_M; i++) {
      /* select random grid node other then S,T */
      x = random_int(2, X*Y-1);
      /* select random extra node */
      y = random_int(X*Y + 1, N);
      if (random_bit())
        printf("a %8d %8d %10d %10d %10d \n",
                x,y,
                0, random_capacity(XDEG), random_cost(1));
      else
        printf("a %8d %8d %10d %10d %10d \n",
                y,x,
                0, random_capacity(XDEG), random_cost(1));
    }
  }
  else
    /* scatter extra arcs in the sceleton graph */
    for (i = 1; i <= EXTRA_M; i++) {
      /* select random grid node other then S,T */
      x = random_int(1, X*Y-1);
      /* select different grid node other then S,T */
      do
        y = random_int(1, X*Y-1);
      while (x == y);
      /* connect x and y */
      printf("a %8d %8d %10d %10d %10d \n",
              x,y,
              0, random_capacity(XDEG), random_cost(1));
    }

  /* return path */
  for (x = 0; x <  X; x++) {
    for (y = 0; y < Y-1; y++) {
      printf("a %8d %8d %10d %10d %10d \n",
              grid_to_id(x, y), grid_to_id(x, y+1),
              0, SUPPLY, RETCOST);
    }
    if (x < X-1)
      printf("a %8d %8d %10d %10d %10d \n",
              grid_to_id(x, Y-1), grid_to_id(x+1, 0),
              0, SUPPLY, RETCOST);
  }
}

long extra_arcs()

{
 
 if (EXTRA_N > 0)
   return(M - (EXTRA_N + 1)             /* to connect extra nodes */
            - X *Y*(XDEG + YDEG)        /* torus arcs */
            - Y*((XDEG*(XDEG+1))/2)     /* cut arcs */
            + 2*XDEG         /* cut arcs into S and out of T */
            - (X*Y - 1));    /* return path */
 else
   return(M - X *Y*(XDEG + YDEG)        /* torus arcs */
            - Y*((XDEG*(XDEG+1))/2)     /* cut arcs */
            + 2*XDEG         /* cut arcs into S and out of T */
            - (X*Y - 1));    /* return path */
            
}


main(argc, argv)
int argc;
char* argv[];

{
  /* get input */

  scanf("%d", &N);
  scanf("%d", &M);
  scanf("%d", &MAXCAP);
  scanf("%d", &MAXCOST);
  scanf("%d", &seed);

  /* check input for correctness */
  if (N < 15) { 
    fprintf(stderr,"ERROR: (# nodes) < 15\n"); 
    exit(3); 
  }
  if (M < 6*N) { 
    fprintf(stderr,"ERROR: (# arcs) < 6*(# nodes)\n"); 
    exit(3); 
  }
  if (M > (long) (pow((double) N, ((double) 5 / (double) 3)))) {
    fprintf(stderr,"ERROR: (#arcs) > (# nodes)^(5/3)\n"); 
    exit(3); 
  }
  if (MAXCAP < 8) {
    fprintf(stderr,"ERROR: MAXCAP < 8\n");
    exit(3); 
  }
  if (MAXCOST < 8) {
    fprintf(stderr,"ERROR: MAXCAP < 8\n");
    exit(3); 
  }

  printf("c Grid example: N = %d M = %d MAXCAP = %d MAXCOST = %d SEED = %d\n",
         N, M, MAXCAP, MAXCOST, seed);


  /* initialize length, width, and remaining nodes */
  Y = 1;
  do Y++; while (Y*Y*Y <= N);
  Y--;
  X = Y*Y;
  do Y++; while (X*Y <= N);
  Y--;
  do X++; while (X*Y <= N);
  X--;

  EXTRA_N = N - X*Y;

  /* initialize degrees and remaining arcs */
  YDEG = 1;
  do {
    YDEG++;
    XDEG = YDEG*YDEG;
  } while (extra_arcs() >= 0);
  YDEG--;
  XDEG = YDEG*YDEG;
      do YDEG++; while (extra_arcs() >= 0);
      YDEG--;
  if (YDEG >= Y) YDEG = Y-1;
  do XDEG++; while (extra_arcs() >= 0);
  XDEG--;
  if (XDEG >= X) XDEG = X-1;
  
  EXTRA_M = extra_arcs();


  if (XDEG >= YDEG)
    MAXDEG = XDEG;
  else
    MAXDEG = YDEG;
  S = 1;
  T = X*Y;

  SMALLCAP = (long) ceil(sqrt((double) MAXCAP));

  /* choose cost of return path arcs */
    RETCOST = MAXCOST / Y;
  if (RETCOST == 0)
    RETCOST == 1;

  SUPPLY = SMALLCAP;

  ALPHA = pow((double) MAXCAP, (double) 1 / (double) (MAXDEG+2));

  /* allocate storage */
    capacity = (long *) calloc(X*Y*(XDEG+YDEG), sizeof(long));
      cost = (long *) calloc(X*Y*(XDEG+YDEG), sizeof(long));

  printf("c X=%d Y=%d XDEG=%d YDEG=%d\n",
         X, Y, XDEG, YDEG);
  printf("p min %d %d\n", N, M);
  build_example(X,Y,XDEG,YDEG);


}


