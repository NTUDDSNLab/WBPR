#include"../include/parallel_graph.cuh"
#include"../include/serial_graph.h"
#include <unistd.h>
#include <iostream>
#include <string.h>

int main(int argc, char **argv)
{
    // Parse command line arguments
    int ch;
    std::string *filename;
    while ((ch = getopt(argc, argv, "f:")) != -1)
    {
        switch (ch)
        {
        case 'f':
            filename = new std::string(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s -f filename\n", argv[0]);
            exit(1);
        }
    }


    ull *V = new ull(0);
    ull *gpu_V;
    ull *E = new ull(0);
    ull *source = new ull(0);
    ull *sink = new ull(0);
    
    // declaring variables to store graph data on host as well as on CUDA device global memory 
    int *cpu_height = NULL,*gpu_height = NULL;
    int *cpu_excess_flow = NULL,*gpu_excess_flow = NULL;
    int *Excess_total = NULL;
    int *cpu_adjmtx = NULL,*gpu_adjmtx = NULL;
    int *cpu_rflowmtx = NULL,*gpu_rflowmtx = NULL;

    readFromDIMACSFormat(*filename, V, E, source, sink, &cpu_height, &cpu_excess_flow, &cpu_adjmtx, &cpu_rflowmtx);

    
    // Print the graph's information
    printf("Reading graph from %s\n", filename->c_str());
    printf("Number of vertices: %llu\n", *V);
    printf("Number of edges: %llu\n", *E);
    printf("Source vertex: %llu\n", *source);
    printf("Sink vertex: %llu\n", *sink);
    
    // allocating host memory
    cpu_height = (int*)malloc(*V*sizeof(int));
    cpu_excess_flow = (int*)malloc(*V*sizeof(int));
    Excess_total = (int*)malloc(sizeof(int));
    


    // allocating CUDA device global memory
    CHECK(cudaMalloc((void**)&gpu_height,*V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_excess_flow,*V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_adjmtx,*V**V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_rflowmtx,*V**V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_V,sizeof(ull)));


    //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // time start

    // invoking the preflow function to initialise values in host
    preflow(V,*source,*sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total);

    // print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // copying host data to CUDA device global memory
    CHECK(cudaMemcpy(gpu_height,cpu_height,*V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_excess_flow,cpu_excess_flow,*V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_adjmtx,cpu_adjmtx,*V**V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,*V**V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_V,V,sizeof(ull),cudaMemcpyHostToDevice));

    // push_relabel()
    push_relabel(V,gpu_V,*source,*sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);
    

    // print values from both implementations
    printf("The maximum flow value of this flow network as calculated by the parallel implementation is %d\n",cpu_excess_flow[*sink]);
    

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
