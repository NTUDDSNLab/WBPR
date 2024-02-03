#include "graphtypes.h"

main(argc,argv)
int argc;
char *argv[];

{   Graph graph;
    int size,i,flow;
    Edge edge;

    graph = ReadGraph(&size,argv[1]);

    MaxFlow(graph,1,size,&flow);

    printf("flow=%d\n",flow);
    for (i=1; i<=size; i++) {
	printf("%3d: ",i);
	for (edge=FirstEdge(graph,i); edge!=NULL; edge=NextEdge(edge))
	    if (ELabel2(edge) > 0)
		printf("%d %d  ",EndPoint(edge),ELabel2(edge));
	printf("\n");
	}

}


