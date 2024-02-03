/* Distance Matrix graph type */

#define MATRIX

typedef MatrixGraph DistGraph;

#define dist(graph,i,j) (graph[(i)*graph[0]+j])
#define ReadDist(size,file) ReadMatrix(size,file)
#define WriteDist(graph,file) WriteMatrix(graph,file)
#define NewDist(size) NewMatrix(size)
#define GraphSize(g) (g[0])


/* tourlib.c definitions */
#define Tourcost(graph,tour) TourcostM(graph,tour)
#define WriteTourPts(graph,tour,file) WriteTourPtsM(graph,tour,file)

/* Graph algorithm definitions */
#define DistPrim(graph,cost) MatrixPrim(graph,cost)
#define DistWeighted_Match(graph,maximize) Weighted_Match(graph,3,maximize)
#define DistDijkstra(graph,node) MatrixDijkstra(graph,node)
#define DistMaxFlow(graph,source,sink,flow) MatrixMaxFlow(graph,source,sink,flow)


