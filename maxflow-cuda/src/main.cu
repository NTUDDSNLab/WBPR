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

    printf("Reading graph from file %s\n",filename);

    if (csr_graph.checkIfContinuous()) {
        printf("Graph is continuous\n");
    } else {
        printf("Graph is not continuous, terminate!\n");
        exit(1);
    }
    
    int V = csr_graph.num_nodes;
    int E = csr_graph.num_edges;


    // declaring variables to store graph data on host as well as on CUDA device global memory 
    int *cpu_height,*gpu_height;
    int *cpu_excess_flow,*gpu_excess_flow;
    int *Excess_total;
    int *cpu_destination,*gpu_destination;
    int *cpu_offsets,*gpu_offsets;
    int *cpu_capcities, *gpu_capcities;
    int *cpu_rflow, *gpu_rflow;

    
    // allocating host memory
    cpu_height = (int*)malloc(V*sizeof(int));
    cpu_excess_flow = (int*)malloc(V*sizeof(int));
    cpu_destination = (int*)malloc(V*sizeof(int));
    cpu_offsets = (int*)malloc(E*sizeof(int));
    cpu_capcities = (int*)malloc(E*sizeof(int));
    // TODO: Construct the residual graph with CSR format
    cpu_rdestination = (int*)malloc(V*sizeof(int));
    cpu_roffsets = (int*)malloc(E*sizeof(int));
    cpu_rflow = (int*)malloc(E*sizeof(int));
    
    
    Excess_total = (int*)malloc(sizeof(int));




    // allocating CUDA device global memory
    cudaMalloc((void**)&gpu_height,V*sizeof(int));
    cudaMalloc((void**)&gpu_excess_flow,V*sizeof(int));
    cudaMalloc((void**)&gpu_destination,V*sizeof(int));
    cudaMalloc((void**)&gpu_offsets,E*sizeof(int));
    cudaMalloc((void**)&gpu_capcities,E*sizeof(int));
    cudaMalloc((void**)&gpu_rflow,E*sizeof(int));


    // readgraph
    // readgraph(filename,V,E,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx);
    

    // time start

    // invoking the preflow function to initialise values in host
    preflow(V,source,sink,cpu_height,cpu_excess_flow,cpu_destination,cpu_offsets,cpu_capcities, Excess_total);
    printf("Excess_total: %d\n",*Excess_total);
    //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // copying host data to CUDA device global memory
    cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_excess_flow,cpu_excess_flow,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_adjmtx,cpu_adjmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);

    // push_relabel()
    push_relabel(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);
    
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
    cudaFree(gpu_height);
    cudaFree(gpu_excess_flow);
    cudaFree(gpu_adjmtx);
    cudaFree(gpu_rflowmtx);
    
    // free host memory
    free(cpu_height);
    free(cpu_excess_flow);
    free(cpu_adjmtx);
    free(cpu_rflowmtx);
    free(Excess_total);
    
    // return 0 and end program
    return 0;

}
