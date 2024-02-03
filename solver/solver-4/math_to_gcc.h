/* Macros for converting from Turboc math.h to gcc math.h */

#ifndef MATH_TO
#define MATH_TO

#include <math.h>

double drand48(void);
void srand48(long int);

#define random(A) (int)(drand48()*(double)(A))

#define sgn(A)  ((A) > 0) ? 1 : ((A)== 0) ? 0 : -1

#endif
