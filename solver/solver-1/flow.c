/* Implementation of "A New Max-Flow Algorithm" by Andrew Goldberg */
/* Implementation suggested by R. Tarjan */
/* Max-Flow in undirected graphs only. */
/* Program written by Edward Rothberg  12/85 */

/* Associated with each vertex is a lower bound on the distance to the sink */
/* and a flow excess. Start with infinite excess at source.  */

/* Stage 1: Do a breadth first search in toward the sink to initalize the
	distance labels.  For each vertex with positive excess and non-infinite 	distance label, push flow excess to adjacent vertices with smaller
	distance labels along edges with remaining capacity.  Do an occasional
	search to keep distance labels current.  Stop when all flow excess is at
	vertices with infinite distance labels. */

/* Stage 2: Push excess flow back to source. */


#include "graphtypes.h"

#define cap(edge) ELabel(edge)		 /* edge capacity */
#define flow(edge) ELabel2(edge)	 /* edge flow */
#define res(edge) (cap(edge)-flow(edge)) /* edge residual capacity */

static int *distance;		/* lower bound on distance to sink */
static int *excess;		/* node's flow excess */
static int source,sink;


struct queue {
    int front, back, size;
    int *list;
    };

static struct queue nodeQ; /* queue for flow algorithm */
			/* used to hold vertices with positive flow excess */
			/* and distance labels less than infinity */

static struct queue searchQ;	/* queue for breadth-first search */


static Graph fgraph;	/* graph we're working with */
static int gsize;


/* Given undirected graph, find the maximum flow from source s to sink t */
/* Returns the graph with the label2 edge field set to the flow in that edge */
/* Note: in this algorithm, label = edge capacity    label2 = edge flow */
/*	a positive flow represents outgoing flow */
/*	a negative flow represents incoming flow */


MaxFlow(graph,s,t,quantity)
Graph graph;
int s,t,*quantity;

{   int i;
    Edge edge;

    source = s;  sink = t;
    fgraph = graph;
    gsize = Degree(graph,0);
    for (i=1; i<=gsize; i++)
	for (edge=FirstEdge(fgraph,i); edge!=NULL; edge=NextEdge(edge))
	    flow(edge) = 0;

    distance = (int *) malloc((gsize+1)*sizeof(int));
    excess = (int *) malloc((gsize+1)*sizeof(int));
    nodeQ.list = (int *) malloc((gsize+1)*sizeof(int));
    searchQ.list = (int *) malloc((gsize+1)*sizeof(int));

    Search(sink);	/* do breadth-first search to get */
			/* initial distances labels */

    for (i=1; i<=gsize; i++)
	excess[i] = 0;
    excess[source] = INF;

    /* initially only source has excess */
    Initqueue(&nodeQ);
    Enqueue(&nodeQ,source);

    Stage1();	/* do Stage 1.  Push excess towards sink */

    *quantity = excess[sink];	/* amount of flow which reaches sink */

/*  do a search to find the min-cut */
/*  Search(sink);
    printf("cut capacity %d\n",excess[sink]);
    printf("cut: ");
    for (i=1; i<=gsize; i++)
	if (distance[i]==INF) printf("%d ",i);
    printf("\n\n");
*/

    Search(source);	/* initialize distance labels for stage 2 */

    excess[source] = 0;
    excess[sink] = 0;
    Initqueue(&nodeQ);
    for (i=1; i<=gsize; i++)
	if (excess[i]>0)
	    Enqueue(&nodeQ,i);	/* put all vertices with positive excess on */
				/* the queue */

    Stage2();	/* do Stage 2.  Push excess back to source */
}


Stage1()
{
    int node;

    while ((node=Dequeue(&nodeQ))!=-1) {
	Pulse(node);
	}
}


/* try to send excess from 'node' out to adjacent vertices. */
/* Note: when sending flow to 'lower' labelled vertices, the distance */
/* of 'node' is never actually consulted.  The minimum distance label */
/* of a reachable adjacent vertex is found, and the distance of 'node' */
/* is necessarily one greater. */

Pulse(node)
int node;
{   int extra,quantity,epoint,level,dosearch;
    Edge edge;


    /* no need to get rid of excess from sink and infinite distance vertices */
    if (node==sink || distance[node]==INF) return;


    extra = excess[node];
    dosearch = 0;

    while (extra > 0) {

	level = Min(node);	/* push to adjacent vertex with smallest */
				/* distance label */
	if (level == INF) break;

	edge = FirstEdge(fgraph,node);
	while (edge != NULL && extra > 0) {

	    /* if flow can be sent down the edge */
	    if (res(edge)>0 && distance[EndPoint(edge)] == level) {

		quantity = res(edge);			/* send as much as */
		if (quantity > extra) quantity = extra; /* can be sent */

		Push(edge,quantity);

		epoint = EndPoint(edge);

		/* put adjacent vertex on the queue if it isn't there already */
		if (excess[epoint] == 0) Enqueue(&nodeQ,epoint);

		/* if we sent flow to the sink, redo distance labels */
		if (epoint==sink) dosearch = 1;

		excess[epoint] += quantity;
		extra -= quantity;
		}
	    edge = NextEdge(edge);
	    }
	}

    if (dosearch) Search(sink);    /* redo all distance labels */

    else {
	if (extra == 0) level = Min(node);    /* update 'node' distance label */

	if (level >= gsize-1) distance[node]=INF;

	else distance[node] = level+1;
	}

    excess[node]=extra;

    /* if we didn't get rid of all the excess, put vertex back on the queue */
    if (extra>0 && distance[node]!=INF) Enqueue(&nodeQ,node);
}


/* find minimum distance label of a vertex which is adjacent to and */
/* reachable from 'i'. */

int Min(i)
int i;
{   Edge edge;
    int min,epoint;

    /* find the smallest distance label of a reachable adjacent vertex */
    edge = FirstEdge(fgraph,i);
    min = INF;
    while (edge!=NULL) {
	if (res(edge)>0) {
	    epoint = EndPoint(edge);
	    if (distance[epoint] < min) min = distance[epoint];
	    }
	edge = NextEdge(edge);
	}
    if (min < gsize) return(min);
    else return(INF);
}


/* push 'quantity' down 'edge' */

Push(edge,quantity)
Edge edge;
int quantity;
{
    flow(edge) += quantity;
    flow(Other(edge)) -= quantity;
}




/* Stage 2: Push excess back to source. */
/* Since we are pushing flow back, we only need to consider edges which */
/* already have flow.  Thus, a vertex is reachable from another only if */
/* it is possible to push existing flow back to it. */


Stage2()
{
    int node;

    while ((node=Dequeue(&nodeQ))!=-1) {
	Pulse2(node);
	}
}

/* same as Pulse 1, except only existing flow can be pushed back */

Pulse2(node)
int node;
{   int extra,quantity,epoint,level,dosearch;
    Edge edge;

    if (node==source || distance[node]==INF) return;

    extra = excess[node];

    dosearch = 0;

    while (extra > 0) {

	level = Min2(node);
	if (level == INF) break;

	edge = FirstEdge(fgraph,node);
	while (edge != NULL && extra > 0) {
	    if (flow(edge)<0 && distance[EndPoint(edge)] == level) {

		quantity = -flow(edge);
		if (quantity > extra) quantity = extra;
		Push(edge,quantity);

		epoint = EndPoint(edge);

		if (excess[epoint] == 0) Enqueue(&nodeQ,epoint);
		if (epoint==source) dosearch = 1;

		excess[epoint] += quantity;
		extra -= quantity;
		}
	    edge = NextEdge(edge);
	    }
	}

    if (dosearch) Search(source);
    else {
	if (extra == 0) level = Min2(node);

	if (level >= gsize-1) distance[node]=INF;

	else distance[node] = level+1;
	}

    excess[node]=extra;
}


int Min2(i)
int i;
{   Edge edge;
    int min,epoint;

    edge = FirstEdge(fgraph,i);
    min = INF;
    while (edge!=NULL) {
	if (flow(edge)<0) {
	    epoint = EndPoint(edge);
	    if (distance[epoint] < min) min = distance[epoint];
	    }
	edge = NextEdge(edge);
	}
    if (min < gsize) return(min);
    else return(INF);
}



/* do a breadth-first search into 'start'.  (i.e. Search(source) starts       */
/* at the source finds the shortest distances from all other vertices into    */
/* the source through edges with residual capacity. */

Search(start)
int start;

{   int i,node,level,epoint;
    Edge edge;

    /* distance is -INF if node has not yet been reached, -1 if it  */
    /* is currently in the queue, and positive if it has been set   */
    /* An INF is placed in the queue to indicate a move to a higher */
    /* search depth. */

    Initqueue(&searchQ);

    for (i=1; i<=gsize; i++)
	distance[i]=-INF;

    Enqueue(&searchQ,start);
    distance[start] = -1;

    Enqueue(&searchQ,INF);
    level=0;

    while (level < gsize) {
	node = Dequeue(&searchQ);
	if (node==INF) {
	    level++;
	    Enqueue(&searchQ,INF);
	    }
	else {
	    distance[node] = level;

	    /* enqueue each vertex adjacent to 'node' which is reachable */

	    for (edge=FirstEdge(fgraph,node); edge!=NULL; edge=NextEdge(edge)) {

		/* if flow = -cap, the incoming edge is saturated, skip it */
		if (flow(edge)==-cap(edge))
			continue;

		epoint = EndPoint(edge);
		if (distance[epoint]==-INF) {
		    distance[epoint] = -1;
		    Enqueue(&searchQ,epoint);
		    }
		}
	    }
	}

    /* any vertex with negative distance label is unreachable from 'start */
    for (i=1; i<=gsize; i++)
	if (distance[i]<0) distance[i]=INF;

}



/* Operations on queue data type */

Initqueue(Q)
struct queue *Q;
{
    Q->size  = gsize;
    Q->front = 0;
    Q->back  = gsize;
}


Enqueue(Q,node)
struct queue *Q;
int node;
{
    if (Q->front==Q->back) { printf("Error: queue full\n"); exit(0); }
    Q->list[Q->front++] = node;
    if (Q->front > Q->size) Q->front = 0;
}


int Dequeue(Q)
struct queue *Q;
{
    Q->back++;
    if (Q->back > Q->size) Q->back = 0;
    if (Q->back == Q->front) {
	return(-1);
	}
    return(Q->list[Q->back]);
}


