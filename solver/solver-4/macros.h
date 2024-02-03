/* These are some useful macro-declarations */


#ifndef MYLIB_
#define MYLIB_

#define MIN3(A,B,C) (((A)<(B)) ? (((A)<(C))?(A):(C)):(((B)<(C))?(B):(C)))

#define MIN2(A,B) (((A)<(B))?(A):(B))

#define MAX3(A,B,C) (((A)>(B)) ? (((A)>(C))?(A):(C)):(((B)>(C))?(B):(C)))

#define MAX2(A,B) (((A)>(B))?(A):(B))

#define sgn(A)  (((A) > 0) ? 1 : ((A)== 0) ? 0 : -1)

#endif

