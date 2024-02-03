#include "graphtypes.h"
#include "matrix.h"

main(argc,argv)
int argc;
char *argv[];

{	DistGraph graph;
	Graph g;
	int size,i,flow;
	Edge edge;

	graph = ReadDist(&size,argv[1]);

	g = DistMaxFlow(graph,1,size,&flow);

	printf("flow=%d\n",flow);
	for (i=1; i<=size; i++) {
		printf("%3d: ",i);
		for (edge=FirstEdge(g,i); edge!=NULL; edge=NextEdge(edge))
			if (ELabel2(edge) > 0)
				printf("%d %d  ",EndPoint(edge),ELabel2(edge));
		printf("\n");
	}

}


