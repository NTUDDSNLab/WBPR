/* grid generator */

#include <stdio.h>

#define MAXCAP  1000  /* maximum edge capacity */ /* used to be 10000 */
#define MAXCOST 1000  /* maximum edge cost */ /* used to be 10000 */

/* global variables */

int X, Y, XDEG, YDEG, seed;
char file_name[20];
FILE *fp;

int random_bit()
{
int b;

b = random()&01;
if (b==1) return(1);
else return(-1);
}

int random_int(max)
int max;
{
return((int) random()%max);
}

int exp2(x)
int x;
{

int ans;

  for (ans = 1; x > 0; x--)
    ans = ans * 2;
  return(ans);
}

int random_capacity(dist) 
int dist;
/* returns random capacity conditioned by dist */

{
int ans;

  ans = random_int(MAXCAP) + 1;
  ans = ans / exp2(dist-1);
  return(ans);

}

int random_cost(dist,u)
int dist,u;
/* returnes random cost conditioned by dist and u */

{
int ans;

  ans = random_bit() * random_int(MAXCOST);
  return(ans);

}
  
int grid_to_id(x, y, x_dim)
int x,y,x_dim;
{

return((y * x_dim) + x);

}

build_example(X,Y,XDEG,YDEG,fp)
int X,Y,XDEG,YDEG;
FILE *fp;

{
int x,y,z,i,l,u,c;

  for (y = 0; y < Y; y++) {
    for (x = 0; x < X; x++) {
      /* x-direction edges */
      for (i = 1; i <= XDEG; i++){
        z = (x+i) % X;
        l = - random_capacity(i);
        u = random_capacity(i);
        c = random_cost(i,u);
        fprintf(fp,"a %8d %8d %10d %10d %10d \n", 
                grid_to_id(x, y, X)+1,
                grid_to_id(z, y, X)+1,
                l, u, c);
      }
      /* y-direction edges */
      for (i = 1; i <= YDEG; i++){
        z = (y+i) % Y;
        l = - random_capacity(i);
        u = random_capacity(i);
        c = random_cost(i,u);
        fprintf(fp,"a %8d %8d %10d %10d %10d \n",
                grid_to_id(x, y, X)+1,
                grid_to_id(x, z, X)+1,
                l, u, c);
      }
    }
  } 
}

main()

{

  /* get info from user */
  printf("          X: ");
  scanf("%d", &X);
  printf("       XDEG: ");
  scanf("%d", &XDEG);
  printf("          Y: ");
  scanf("%d", &Y);
  printf("       YDEG: ");
  scanf("%d", &YDEG);

  printf("       Seed: ");
  scanf("%d", &seed);
  srandom(seed);

  printf("Output file: ");
  scanf("%s",file_name);
  fp = fopen(file_name,"w");

  fprintf(fp,"c Grid example: X=%d, Y=%d, XDEG=%d, YDEG=%d, SEED=%d.\n",
           X, Y, XDEG, YDEG, seed);
  fprintf(fp,"p min %d %d\n", X*Y, X*Y*(XDEG + YDEG));
  build_example(X,Y,XDEG,YDEG,fp);

  fclose(fp);

}

