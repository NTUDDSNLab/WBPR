#include "graph.h"


int main(int argc, char** argv) {
    printf("Reading graph from binary file %s\n",argv[1]);
    CSRGraph graph;
    graph.loadFromBinary(argv[1]);
    int source  = atoi(argv[2]);
    int sink    = atoi(argv[3]);

    printf("Building residual graph\n");

    ResidualGraph rgraph;
    rgraph.buildFromCSRGraph(graph);
    printf("Finished building residual graph\n");
    //rgraph.print();
    


    printf("Computing maxflow from %d to %d\n",source,sink);
    rgraph.maxflow(source, sink);

    return 0;
}