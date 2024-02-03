/* netio.h == header for netio.c */

#ifndef _NETIO_H
#define _NETIO_H


network * read_net(char * filename);

void free_net(network * n);

void print_net(FILE * output, network * n, char * s, int quiet
			   , char * prob_name, controll * cont);  
                                        /* t = title
										   p = parameter settings
										   s = DIMACS std output
										   
										/  a = all adjacency lists 
          only if compiled with DEBUG  {   e = edges
										\  v = vertices */
void print_DIMACS_std(FILE * output, network * n, char * prob_name);

#ifdef DEBUG
void print_vertex(vertex * v, int quiet);

void print_edge(edge * e, int quiet);
#endif

void print_err(char * message, int stop);
#endif
