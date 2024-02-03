/* Data types and global declarations 
   for the Cheriyan-Hagerup Max-flow algorithm */

/* Copyright:
	This program was written by 

	Tamas Badics, 1991,
	Rutgers University, RUTCOR
	P.O.Box 5062
	New Brunswick, NJ, 08903
	e-mail: badics@rutcor.rutgers.edu
 
	The code may be used and modified for not-for-profit use.
	This notice must be remained.	
====================================================================*/


#ifndef _MAXFLOW_H
#define _MAXFLOW_H


/*==================================================================*/
/*========  NETWORK type for the algorithm  ========================*/
/*========  see maxflow_typedef.h           ========================*/
/*==================================================================*/

#include <stdio.h>
#include "maxflow_typedef.h"

/*==================================================================*/
/*======  Global variables          ================================*/
/*==================================================================*/
extern int Itnum;

/*==================================================================*/
/*========   Function declarations   ===============================*/
/*==================================================================*/

int maxflow_cold_start(char * prob_file, network ** n,controll * cont);
                           /* Reads the problem from the file.
							  Then start the PLED algorithm.
							  Return false if infeasible.
							  Return the network via n.
							  See maxflow_typedef.h for controll.*/

int maxflow_warm_start(network * n, int need_preflow, controll * cont);
	                          /* After the allocations, with
								 a prepared network * n it will
								 start the maxflow algorithm.
								 Return false if infeasible.
								 if (need_preflow == true) then 
								 it will make a preflow before
								 starting. Otherwise starting preflow
								 is assumed.*/

int maxflow(network * n, controll * cont);
                              /* Find the max-flow in the initialized 
								 network n using the PLED alg. 
								 Return the # of iterations*/

int check_maxflow(network *n);
#endif


