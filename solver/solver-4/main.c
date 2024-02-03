/* Main routine for the PLED algorithm */
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


#include <stdio.h>
#include "maxflow.h"
#include "netio.h"

#ifdef DEBUG
#include "debug.h"
#endif

int look_up(char * s);
void print_usage(void);

void main(int argc, char * argv[])
{
	network * n;
	char * report;
	int i, feas, quiet;
	controll cont;
	char * prob_file;
	FILE * output;
	
	output = stdout;
/* !!!!! should be report = "s" !!!!! */
	report = "";
	cont.perm = 0;
	cont.print_freq = 0;
	cont.stall_freq = -1;
	cont.relab_freq = -1;
	cont.cut_freq = -1;
	cont.gap = -1;
	cont.quiet = 0;

	prob_file= NULL;
	
	for (i = 1; i < argc; i++){
		switch (look_up(argv[i])){
		  case 0: 
			prob_file= argv[++i];
			break;
		  case 1: 
			cont.print_freq = atoi(argv[++i]);
			break;
		  case 2: 
			cont.stall_freq = atoi(argv[++i]);
			break;
		  case 3: 
			cont.relab_freq = atoi(argv[++i]);
			break;
		  case 4: 
			report = argv[++i];
			break;
		  case 5: /*help*/
		  case 6:
			print_usage();
			break;
		  case 7:
			output = fopen(argv[++i],"w");
			if (output == NULL) {
				fprintf(stderr
                    ,"PLED: Output file %s can't be opened\n",argv[i]);
				exit(0);
			}	
			break;
		  case 8:
			cont.quiet = 1;
			break;
		  case 9:
			cont.perm = 1;
			break;
		  case 10: 
			cont.cut_freq = atoi(argv[++i]);
			break;
		  case 11:
			cont.gap = 0;
			break;
		  default:
			break;
		} 
	}

	quiet = cont.quiet;
	
	if (prob_file == NULL && !quiet)
	  fprintf(stderr, "Input for PLED is coming from stdin!\n");
	
	feas = maxflow_cold_start(prob_file, &n, &cont);

	if (!quiet){ 
		if (feas){
			printf("\nIteration finished at %d\n\n", Itnum);
			if (check_maxflow(n)){ 
				printf("Maxflow is correct.\n");
			}else{ 
				print_err("Err: Wrong maxflow value!!!!", 1);
			}
			printf("The maximum flow: %d\n", (int)(n->maxflow));
		}else{ 
			printf("NO FEASIBLE SOLUTION\n");
		}
	}
	
	print_net(output, n, report, quiet, prob_file, &cont);

	free_net(n);

	if (output != stdout)
	  fclose(output);
	
	if (!quiet)
	  printf("END\n");
}
/*=================================================================*/
#define OPS_NUM 12

int look_up(char * s)
{
	char * ops[OPS_NUM] 
	  = { "-in", "-pf", "-sf", "-rf", "-rep"
			, "-h", "-help", "-out", "-quiet"
			  , "-perm", "-cf", "-nogap"};
	int i;
	
	for (i = 0; i < OPS_NUM; i++){
		if (strcmp(ops[i], s) == 0)
		  return i;
	} 
	return -1;
} 

void print_usage(void)
{
	printf("Usage: PLED [-in input_file] [-out out_file]\n");
	printf("            [-pf print_freq] [-sf stall_freq]\n");
	printf("            [-rf relabel_freq] [-cf cut_freq]\n");
	printf("            [-rep s|p|t|v|e|a (report_controll)]\n");
	printf("            [-nogap] [-perm] [-quiet]\n");
	exit(0);
}

