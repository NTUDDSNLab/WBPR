/* random.c -- functions dealing with randomization.

	RandomPermutation
        RandomInteger
        InitRandom
	RandomSubset
*/

#include <sys/time.h>


/* RandomPermutation -- contruct a random permutation of the array perm.  
   It is assumed that the length of perm is n.  The algorithm used makes
   a pass through the array, randomly switching elements of the array.
*/
RandomPermutation (perm, n)
int perm[], n;
{
    int i, j, t;
			
    for (i = 0; i < n - 1; i++){
        j = RandomInteger(i, n-1);	/* Swap the element perm[i] with   */
	t = perm[i];			/* a random element from the range */
	perm[i] = perm[j];		/* i..n-1.			   */ 
	perm[j] = t;
    }
}

RandPerm (perm, n)
int perm[], n;
{
    int i, j, t;
	
    for (i = 0; i < n; i++)
        perm[i] = i;

    for (i = 0; i < n - 1; i++){
        j = RandomInteger(i, n-1);	/* Swap the element perm[i] with   */
	t = perm[i];			/* a random element from the range */
	perm[i] = perm[j];		/* i..n-1.			   */ 
	perm[j] = t;
    }
}



/* RandomInteger -- return a random integer from the range low .. high.
*/
int RandomInteger (low, high)
int low, high;
{
    return random() % (high - low + 1) + low;
}

/* InitRandom -- If the seed is non-zero, the random number generator is
   initialized with the seed, giving a fixed sequence of "random" numbers.
   If the seed is zero, then the time of day is used to intialize the random 
   number generator. 
*/
InitRandom (seed)
int seed;
{
    struct timeval tp;
   
    if (seed == 0){
        gettimeofday(&tp, 0);
        srandom(tp.tv_sec + tp.tv_usec);
    }
    else
	srandom(seed);
}

/* RandomSubset - return n distinct values, randomly selected between
high and low, the algorithm is inefficient if n is large - this could
be improved */
RandomSubset(low, high, n, x)
int low, high, n, *x;
{
  int i, j, r, flag;

  if (high - low + 1 < n)
    Barf("Invalid range for Random Subset");

  i = 0;
  while (i < n){
    r = RandomInteger(low, high);
    flag = 0;
    for (j = 0; j < i; j++)
      if (x[j] == r)
	flag = 1;
    if (flag == 0)
      x[i++] = r;
  }
  
}






