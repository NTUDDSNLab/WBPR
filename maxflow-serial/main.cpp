#include "graph.h"
#include <unistd.h>



int main(int argc, char** argv) {

    CSRGraph graph;
    /* Argparse */
    int ch;
    int file_type = 0;
    bool auto_tune = false;
    int source = -1;
    int sink = -1;
    char* filename = NULL;

    while ((ch = getopt(argc, argv, "hv:f:s:t:a")) != -1) {
        switch(ch)
        {
            case 'h':
                printf("Usage: %s [-h] [-v file type] [-f filename] [-s source] [-t sink]\n",argv[0]);
                printf("Options:\n");
                printf("\t-h\t\tShow this help message and exit\n");
                printf("\t-v\t\tSpecify which kind of file to load\n");
                printf("\t\t\t\t0: txt file (default)\n");
                printf("\t\t\t\t1: binary file\n");
                printf("\t\t\t\t2: DIMACS file\n");
                printf("\t-f filename\tSpecify the file path (binary or txt)\n");
                printf("\t-s source\tSource node\n");
                printf("\t-t sink\t\tSink node\n");
                exit(0);
                break;
            case 'v':
                file_type = atoi(optarg);
                PRINTF("File type: %d\n",file_type);
                break;
            case 'f':
                filename = optarg;
                switch(file_type) {
                    case 0:
                        PRINTF("Loading txt file\n");
                        graph.buildFromTxtFile(filename);
                        break;
                    case 1:
                        PRINTF("Loading binary file\n");
                        graph.loadFromBinary(filename);
                        break;
                    case 2:
                        PRINTF("Loading DIMACS file\n");
                        graph.buildFromDIMACSFile(filename);
                        break;
                    default:
                        printf("Unknown file type\n");
                        exit(1);
                }
                break;
            case 's':
                source = atoi(optarg);
                PRINTF("Source: %d\n",source);
                break;
            case 't':
                sink = atoi(optarg);
                PRINTF("Sink: %d\n",sink);
                break;
            case 'a':
                PRINTF("Autotuning mode\n");
                auto_tune = true;
                break;
        }
    }

    if (source == -1 || sink == -1) {
        printf("Source(-s) and sink(-t) must be specified\n");
        exit(1);
    }

    PRINTF("Building residual graph\n");

    ResidualGraph rgraph;
    rgraph.buildFromCSRGraph(graph);
    PRINTF("Finished building residual graph\n");
    rgraph.print();
    
    if (!auto_tune) {
        printf("Starting push-relabel on %s\n", filename);

        if (file_type == 0 || file_type == 1) {
            // If the graph is from txt or binary file, we need to set source and sink
            printf("Source: %d Sink: %d\n",source,sink);
            rgraph.source = source;
            rgraph.sink = sink;
            
            rgraph.maxflow(source, sink);
        } else {
            // If the graph is from DIMACS file, we get the source and sink from the file
            rgraph.source = graph.source_node;
            rgraph.sink = graph.sink_node;
            printf("Source: %d Sink: %d\n",rgraph.source,rgraph.sink);
            rgraph.maxflow(rgraph.source, rgraph.sink);
        }
        

    } else {
        /* Check all pair of nodes (0,1) to (num_node-2, num_node-1) has path */
        printf("Checking all pair of nodes has path...\n");
        bool has_path = false;
        
        // NOTE: For any edges (i, j), i > j 
        for (int i = graph.num_nodes; i > 0; i--) {
            /* If node i has no outgoing neighbor, pass */
            if (graph.offsets[i] == graph.offsets[i-1]) {
                continue;
            }

            for (int j = 0; j < graph.num_nodes; j++) {
                rgraph.source = i;
                rgraph.sink = j;
                if (rgraph.checkPath()) {
                    printf("(%d,%d) has path\n",i,j);
                    has_path = true;
                }
            }
        }
        if (!has_path) {
            printf("No path found\n");
            exit(1);
        }
    }

}