#include"../include/parallel_graph.cuh"
#include"../include/serial_graph.h"
#include "../include/graph.h"

int main(int argc, char **argv)
{
    // checking if sufficient number of arguments (4) are passed in CLI
    if(argc != 4)
    {
        printf("Invalid number of arguments passed during execution\n");
        exit(0);
    }
    // reading the arguments passed in CLI
    char* filename = argv[1];
    // int V = atoi(argv[2]);
    // int E = atoi(argv[3]);
    int source = atoi(argv[2]);
    int sink = atoi(argv[3]);

    // Read from snap txt
    CSRGraph csr_graph;
    csr_graph.buildFromTxtFile(filename);

    ResidualGraph res_graph;
    res_graph.buildFromCSRGraph(csr_graph);

    printf("Reading graph from file %s\n",filename);
    
    int V = csr_graph.num_nodes;
    int E = csr_graph.num_edges;

    // Print res_graph
    //res_graph.print();


    // declaring variables to store graph data on host as well as on CUDA device global memory 
    int *cpu_height,*gpu_height;
    int *cpu_excess_flow,*gpu_excess_flow;
    int *Excess_total;
    int *gpu_destinations, *gpu_rdestinations;
    int *gpu_offsets, *gpu_roffsets;
    int *gpu_capcities;
    int *gpu_flows, *gpu_rflows;

    
    // allocating host memory
    cpu_height = (int*)malloc(V*sizeof(int));
    cpu_excess_flow = (int*)malloc(V*sizeof(int));
    Excess_total = (int*)malloc(sizeof(int));





    // allocating CUDA device global memory
    CHECK(cudaMalloc((void**)&gpu_height, V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_excess_flow, V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_destinations,E*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_offsets, (V+1)*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_capcities, E*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_flows, E*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_rdestinations,E*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_roffsets, (V+1)*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_rflows, E*sizeof(int)));


    // readgraph
    // readgraph(filename,V,E,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx);
    

    // time start
    printf("Starting preflow\n");

    // invoking the preflow function to initialise values in host
    preflow(V,source,sink,cpu_height,cpu_excess_flow, 
            (res_graph.offsets), (res_graph.destinations), (res_graph.capacities), (res_graph.flows),
            (res_graph.roffsets), (res_graph.rdestinations), (res_graph.rflows), Excess_total);
    
    printf("Excess_total: %d\n",*Excess_total);

    // Print the result of preflow,
    printf("Preflow result:\n");
    printf("Flow: ");
    for (int i=0; i < res_graph.num_edges; i++) {
        printf("%d ", res_graph.flows[i]);
    }
    printf("\n");
    printf("Rflow: ");
    for (int i=0; i < res_graph.num_edges; i++) {
        printf("%d ", res_graph.rflows[i]);
    }
    printf("\n");
    


    //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // copying host data to CUDA device global memory
    CHECK(cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_excess_flow,cpu_excess_flow,V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_offsets, res_graph.offsets, (res_graph.num_nodes + 1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_destinations, res_graph.destinations, res_graph.num_edges*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_capcities, res_graph.capacities, res_graph.num_edges*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_flows, res_graph.flows, res_graph.num_edges*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_roffsets, res_graph.roffsets, (res_graph.num_nodes + 1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_rdestinations, res_graph.rdestinations, res_graph.num_edges*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_rflows, res_graph.rflows, res_graph.num_edges*sizeof(int), cudaMemcpyHostToDevice));

    //cudaMemcpy(gpu_adjmtx,cpu_adjmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);
    // cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);

    // push_relabel()
    push_relabel(V,E,source,sink,cpu_height,cpu_excess_flow, 
                res_graph.offsets, res_graph.destinations, res_graph.capacities, res_graph.flows, 
                res_graph.roffsets, res_graph.rdestinations, res_graph.rflows,
                Excess_total,
                gpu_height, gpu_excess_flow,
                gpu_offsets, gpu_destinations, gpu_capcities, gpu_flows,
                gpu_roffsets, gpu_rdestinations, gpu_rflows);
    
    // store value from serial implementation
    int serial_check = check(V,E,source,sink);

    // print values from both implementations
    printf("The maximum flow value of this flow network as calculated by the parallel implementation is %d\n",cpu_excess_flow[sink]);
    printf("The maximum flow of this flow network as calculated by the serial implementation is %d\n",serial_check);
    
    // print correctness check result
    if(cpu_excess_flow[sink] == serial_check)
    {
        printf("Passed correctness check\n");
    }
    else
    {
        printf("Failed correctness check\n");
    }

    // free device memory
    CHECK(cudaFree(gpu_height));
    CHECK(cudaFree(gpu_excess_flow));
    CHECK(cudaFree(gpu_offsets));
    CHECK(cudaFree(gpu_destinations));
    CHECK(cudaFree(gpu_capcities));
    CHECK(cudaFree(gpu_flows));
    CHECK(cudaFree(gpu_roffsets));
    CHECK(cudaFree(gpu_rdestinations));
    CHECK(cudaFree(gpu_rflows));

    //cudaFree(gpu_adjmtx);
    //cudaFree(gpu_rflowmtx);
    
    // free host memory
    free(cpu_height);
    free(cpu_excess_flow);
    free(Excess_total);
    //free(cpu_adjmtx);
    //free(cpu_rflowmtx);
    
    // return 0 and end program
    return 0;

}
