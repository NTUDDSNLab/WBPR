#include"../include/parallel_graph.cuh"

void push_relabel(ull *V, ull *gpu_V, ull source, ull sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx, int *gpu_rflowmtx)
{
    /* Instead of checking for overflowing vertices(as in the sequential push relabel),
     * sum of excess flow values of sink and source are compared against Excess_total 
     * If the sum is lesser than Excess_total, 
     * it means that there is atleast one more vertex with excess flow > 0, apart from source and sink
     */

    /* declaring the mark and scan boolean arrays used in the global_relabel routine outside the while loop 
     * This is not to lose the mark values if it goes out of scope and gets redeclared in the next iteration 
     */

    bool *mark,*scanned;
    mark = (bool*)malloc(*V*sizeof(bool));
    scanned = (bool*)malloc(*V*sizeof(bool));

    // initialising mark values to false for all nodes
    for(ull i = 0; i < *V; i++)
    {
        mark[i] = false;
    }

    while((cpu_excess_flow[source] + cpu_excess_flow[sink]) < *Excess_total)
    {
        // printf("cpu_excess_flow[source]: %d, cpu_excess_flow[sink]: %d\n",cpu_excess_flow[source], cpu_excess_flow[sink]);
        // copying height values to CUDA device global memory
        CHECK(cudaMemcpy(gpu_height,cpu_height,*V*sizeof(int),cudaMemcpyHostToDevice));

        printf("Invoking kernel\n");

        // invoking the push_relabel_kernel
        push_relabel_kernel<<<1,2>>>(gpu_V,source,sink,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);

        cudaDeviceSynchronize();

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            // Handle error
            fprintf(stderr, "Kernel launching error: %s\n", cudaGetErrorString(error));
        }

        // copying height, excess flow and residual flow values from device to host memory
        CHECK(cudaMemcpy(cpu_height,gpu_height,*V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_excess_flow,gpu_excess_flow,*V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_rflowmtx,gpu_rflowmtx,*V**V*sizeof(int),cudaMemcpyDeviceToHost));
        
        printf("After invoking\n");
        //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);
        // perform the global_relabel routine on host
        global_relabel(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,mark,scanned);

        printf("\nAfter global relabel\n");
        //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);
    }

}
