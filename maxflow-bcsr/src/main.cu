#include"../include/parallel_graph.cuh"
#include"../include/serial_graph.h"
#include "../include/graph.h"
#include "../include/utils.cuh"
#include <unistd.h>

int V;
int E;

#ifdef TIME_BREAKDOWN
__managed__ unsigned long long duration[totalThreads * PROFILE_NUM] = {0};
__managed__ unsigned long long totalDuration0 = 0;
__managed__ unsigned long long totalDuration1 = 0;
#endif /* TIME_BREAKDOWN */



int main(int argc, char **argv)
{
    CSRGraph csr_graph;
    int ch;
    int file_type = 0;
    int algo_type = 0;
    int source = -1;
    int sink = -1;
    char* filename = NULL;

    while ((ch = getopt(argc, argv, "hv:f:s:t:a:")) != -1) {
        switch(ch)
        {
            case 'h':
                printf("Usage: %s [-h] [-v file type] [-f filename] [-s source] [-t sink]\n",argv[0]);
                printf("Options:\n");
                printf("\t-h\t\tShow this help message and exit\n");
                printf("\t-v\t\tSpecify which kind of file to load\n");
                printf("\t\t\t\t0: SNAP txt file (default)\n");
                printf("\t\t\t\t1: SNAP binary file\n");
                printf("\t\t\t\t2: DIMACS file\n");
                printf("\t-f filename\tSpecify the file path (binary or txt)\n");
                printf("\t-s source\tSource node\n");
                printf("\t-t sink\t\tSink node\n");
                printf("\t-a algorithm\tSpecify the algorithm to use\n");
                printf("\t\t\t\t0: Thread-centric push-relabel (default)\n");
                printf("\t\t\t\t1: Vertex-centric push-relabel\n");
                exit(0);
            case 'v':
                file_type = atoi(optarg);
                break;
            case 'f':
                filename = optarg;
                switch(file_type) {
                    case 0:
                        printf("Loading from txt file: %s\n", filename);
                        // FIXME: Add support for SNAP txt file
                        printf("Only support DIMACS file for now\n");
                        exit(1);
                        csr_graph.buildFromTxtFile(filename);
                        break;
                    case 1:
                        printf("Loading from binary file: %s\n", filename);
                        // FIXME: Add support for SNAP binary file
                        printf("Only support DIMACS file for now\n");
                        exit(1);
                        csr_graph.loadFromBinary(filename);
                        break;
                    case 2:
                        printf("Loading from DIMACS file: %s\n", filename);
                        csr_graph.buildFromDIMACSFile(filename);
                        break;
                    default:
                        printf("Invalid file type\n");
                        exit(1);
                }             
                break;
            case 's':
                if (file_type != 2) {
                    csr_graph.source_node = atoi(optarg);
                }
                break;
            case 't':
                if (file_type != 2) {
                    csr_graph.sink_node = atoi(optarg);
                }
                break;
            case 'a':
                algo_type = atoi(optarg);
                break;
            default:
                printf("Invalid option\n");
                exit(1);

        }
    }

    // If the input file is DIMACS, the source and sink nodes are the first and last nodes
    if (file_type == 2) {
        source = csr_graph.source_node;
        sink = csr_graph.sink_node;
    }

    
    V = csr_graph.num_nodes;
    E = csr_graph.num_edges;
    source = csr_graph.source_node;
    sink = csr_graph.sink_node;

    // Print the graph information, including the statistics
    csr_graph.printGraphStatus();

    if (algo_type == 0) {

        printf("Using thread-centric push-relabel algorithm\n");
    } else if (algo_type == 1) {
        printf("Using vertex-centric push-relabel algorithm\n");
    } else {
        printf("Invalid algorithm type\n");
        exit(1);
    }

    // Create residual graph
    ResidualGraph res_graph;
    res_graph.buildFromCSRGraph(csr_graph);

    // Print res_graph
    // res_graph.print();
    // exit(0);


    // declaring variables to store graph data on host as well as on CUDA device global memory 
    int *cpu_height,*gpu_height;
    int *cpu_excess_flow,*gpu_excess_flow;
    int *Excess_total;
    int *gpu_destinations;
    int *gpu_offsets;
    int *gpu_capcities;
    int *gpu_fflows; // Forward and backward flows
    int *cpu_avq, *gpu_avq;
    int cycle = csr_graph.num_nodes;
    int *gpu_cycle;


    
    // allocating host memory
    cpu_height = (int*)malloc(V*sizeof(int));
    cpu_excess_flow = (int*)malloc(V*sizeof(int));
    Excess_total = (int*)malloc(sizeof(int));
    cpu_avq = (int*)malloc(V*sizeof(int));

    printf("V: %d, E: %d\n", V, E);
    printf("Initializing cpu_avq\n");
    for (int i = 0; i < V; i++)
    {
        cpu_avq[i] = 0;
    }





    // allocating CUDA device global memory
    cudaMalloc((void**)&gpu_height, V*sizeof(int));
    CHECK(cudaMalloc((void**)&gpu_excess_flow, V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_destinations,E*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_offsets, (V+1)*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_capcities, E*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_fflows, E*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_avq, V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_cycle, sizeof(int)));


    // readgraph
    // readgraph(filename,V,E,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx);
    

    // time start
    printf("Starting preflow\n");

    // invoking the preflow function to initialise values in host
    preflow(V,source,sink,cpu_height,cpu_excess_flow, 
            (res_graph.offsets), (res_graph.destinations), (res_graph.capacities), (res_graph.forward_flows),
            Excess_total);
    
    printf("Excess_total: %d\n",*Excess_total);


    //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // copying host data to CUDA device global memory
    CHECK(cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_excess_flow,cpu_excess_flow,V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_offsets, res_graph.offsets, (res_graph.num_nodes + 1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_destinations, res_graph.destinations, res_graph.num_edges*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_capcities, res_graph.capacities, res_graph.num_edges*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_fflows, res_graph.forward_flows, res_graph.num_edges*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_avq, cpu_avq, res_graph.num_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_cycle, &cycle, sizeof(int), cudaMemcpyHostToDevice));
    //cudaMemcpy(gpu_adjmtx,cpu_adjmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);
    // cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);

    printf("Starting push_relabel\n");

    // push_relabel()
    push_relabel(algo_type, V,E,source,sink,cpu_height,cpu_excess_flow, 
                res_graph.offsets, res_graph.destinations, res_graph.capacities, res_graph.forward_flows, 
                Excess_total,
                gpu_height, gpu_excess_flow,
                gpu_offsets, gpu_destinations, gpu_capcities, gpu_fflows, gpu_avq, gpu_cycle);
    

    // print values from both implementations
    printf("The maximum flow value of this flow network as calculated by the parallel implementation is %d, %d\n",cpu_excess_flow[sink], *Excess_total);
    //printf("The maximum flow of this flow network as calculated by the serial implementation is %d\n",serial_check);
    
    // print correctness check result
    // if(cpu_excess_flow[sink] == serial_check)
    // {
    //     printf("Passed correctness check\n");
    // }
    // else
    // {
    //     printf("Failed correctness check\n");
    // }

    // free device memory
    CHECK(cudaFree(gpu_height));
    CHECK(cudaFree(gpu_excess_flow));
    CHECK(cudaFree(gpu_offsets));
    CHECK(cudaFree(gpu_destinations));
    CHECK(cudaFree(gpu_capcities));
    CHECK(cudaFree(gpu_fflows));
    CHECK(cudaFree(gpu_avq));

    //cudaFree(gpu_adjmtx);
    //cudaFree(gpu_rflowmtx);
    
    // free host memory
    free(cpu_height);
    free(cpu_excess_flow);
    free(Excess_total);
    free(cpu_avq);
    //free(cpu_adjmtx);
    //free(cpu_rflowmtx);
    
    // return 0 and end program
    return 0;

}
