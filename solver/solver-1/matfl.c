#include "graphtypes.h"
#include "matrix.h"

Graph MatrixMaxFlow(graph,source,sink,flow)
DistGraph graph;
int source,sink,*flow;

{	int size, i, j;
	Graph g;
	Edge edge;
	DistGraph output;

	size = GraphSize(graph);
	g = NewGraph(size);

	for (i=1; i<=size; i++) {
		for (j=i+1; j<=size; j++) {
			AddEdge(g,i,j,dist(graph,i,j));
			}
		}
	MaxFlow(g,source,sink,flow);

	return(g);
}


