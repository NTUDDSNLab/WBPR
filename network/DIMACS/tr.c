/*
 TRANSIT GRID NETWORK GENERATOR FOR MAX-FLOW/ (by G. Waissi)
 TWO-WAY SYSTEM             (revised 11/25/90)
                            (revised 01/05/91)
			    (rewritten in C and modified by J. Setubal,
                           July 91. Program changed to generate instances
                           for DIMACS algorithm implementation challenge only.)

 usage: tr <num_nodes> <max-capacity> <seed>
 graph written to standard output

*/

#include <stdio.h>
#include <math.h>

int cap;
int i,z,num,col,imax,zmax,colmax,capacity;
int num_nodes,num_arcs;
int source,sink;
int seed;

main(argc, argv)
int argc;
char *argv[];
{

  imax = atoi(argv[1]);
  capacity = atoi(argv[2]);
  seed = atoi(argv[3]);
  srandom(seed);
  UserValues();
  TransitTwoNet();
}

Gen1()
{
  i = 1;
  for (z = 2; z <= (colmax+1); z++)
    {
      DoArc(i, z);
    }
}

Gen2()
{
  i = imax-colmax+2;
  for (z = zmax+colmax+1;  z <= zmax+2*colmax; z++)
    {
      DoArc(i, imax+2);
      i++;
    }
}

DoArc(i1, k2)
int i1, k2;
{
  if (i1 > num_nodes || k2 > num_nodes)
    {
      printf("DoArc:Error: vertex out of range\n");
      exit(1);
    }
  cap = RandomInteger(capacity);
  printf("a %d %d %d\n",i1,k2,cap);
  cap = RandomInteger(capacity);
  printf("a %d %d %d\n",k2,i1,cap);
}

GenArcs()
{
  int j;

  col = 1;
  num = colmax;
  i = 2;
  do
    {
      do
	{
	  if (i < num+1) DoArc(i, i+1);
	  DoArc(i, i+colmax);
	  i++;
	}
      while (i != num+2);
      num += colmax;
      col++;
    }
  while (col != colmax);
  do
    {
      DoArc(i, i+1);
      i++;
    }
  while (i != imax+1);
}

TransitTwoNet()
{
  Gen1();
  GenArcs();
  Gen2();
}

UserValues()
{
  colmax = (int) (sqrt((double)imax) + 0.5);
  zmax = 2*(imax-colmax);
  num_nodes = imax+2;
  num_arcs = 2*(zmax+2*colmax);
  source = 1;
  sink = num_nodes;
  printf("c Two-Way Transit Grid Network\n");
  printf("c for Max-Flow\n");
  printf("p max %d %d\n",num_nodes,num_arcs);
  printf("n %d s\n",source);
  printf("n %d t\n",sink);
  printf("s %d\n", seed);
  printf("colmax = %d\n", colmax);
}

/* RandomInteger -- return a random integer from the range 1 .. high.
*/
int RandomInteger(high)
int high;
{
    return (random() % high) + 1;
}
