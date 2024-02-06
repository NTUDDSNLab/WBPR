#include"../include/parallel_graph.cuh"
#include "../include/utils.cuh"

bool compare_excess_flow(int *new_excess_flow, int *old_excess_flow, int V)
{
    for(int i = 0; i < V; i++)
    {
        if (new_excess_flow[i] != old_excess_flow[i])
        {
            return false;
        }
    }
    return true;
}

void copy_excess_flow(int *new_excess_flow, int *old_excess_flow, int V)
{
    for(int i = 0; i < V; i++)
    {
        old_excess_flow[i] = new_excess_flow[i];
    }
}

void printExcessFlow(int V, int *excess_flow)
{
    printf("Excess flow values : \n");
    for(int i = 0; i < V; i++)
    {
        printf("%d ",excess_flow[i]);
    }
    printf("\n");
}


void push_relabel(int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows,
                int* cpu_roffsets, int* cpu_rdestinations, int* cpu_flow_idx,
                int *Excess_total, 
                int *gpu_height, int *gpu_excess_flow, 
                int *gpu_offsets, int* gpu_destinations, int* gpu_capacities, int* gpu_fflows, int* gpu_bflows,
                int* gpu_roffsets, int* gpu_rdestinations, int* gpu_flow_idx)
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
    mark = (bool*)malloc(V*sizeof(bool));
    scanned = (bool*)malloc(V*sizeof(bool));

    CudaTimer timer;
    float totalMilliseconds = 0.0f;

    // initialising mark values to false for all nodes
    for(int i = 0; i < V; i++)
    {
        mark[i] = false;
    }

    // FIXME: The Excess_total and cpu_excess_flow[sink] are not converged
    while((cpu_excess_flow[source] + cpu_excess_flow[sink]) < *Excess_total)
    {
        printf("cpu_excess_flow[source]: %d, cpu_excess_flow[sink]: %d\n",cpu_excess_flow[source], cpu_excess_flow[sink]);
        printf("gpu_excess_flow[source]: %d, gpu_excess_flow[sink]: %d\n",gpu_excess_flow[source], gpu_excess_flow[sink]);
        // copying height values to CUDA device global memory
        CHECK(cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_excess_flow, cpu_excess_flow, V*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_fflows, cpu_fflows, E*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_bflows, cpu_bflows, E*sizeof(int), cudaMemcpyHostToDevice));


        printf("Invoking kernel\n");
        
        timer.start();
        // invoking the push_relabel_kernel
        push_relabel_kernel<<<number_of_blocks_nodes,threads_per_block>>>
                (V,source,sink,gpu_height,gpu_excess_flow,
                gpu_offsets,gpu_destinations,gpu_capacities,gpu_fflows,gpu_bflows,
                gpu_roffsets,gpu_rdestinations,gpu_flow_idx);
        cudaDeviceSynchronize();
        timer.stop();
        totalMilliseconds += timer.elapsed();



        printf("Kernel invoked\n");

        // copying height, excess flow and residual flow values from device to host memory
        CHECK(cudaMemcpy(cpu_height,gpu_height,V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_excess_flow,gpu_excess_flow,V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_fflows,gpu_fflows, E*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_bflows,gpu_bflows, E*sizeof(int),cudaMemcpyDeviceToHost));


        // printf("Before global relabel--------------------\n");
        printf("Excess total : %d\n",*Excess_total);
        // printExcessFlow(V,cpu_excess_flow);
        //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        //printf("Excess total : %d\n",*Excess_total);
        // perform the global_relabel routine on host
        // printf("Before global relabel, Excess total : %d\n",*Excess_total);

        global_relabel(V, E, source,sink,cpu_height,cpu_excess_flow,
                      cpu_offsets,cpu_destinations, cpu_capacities, cpu_fflows, cpu_bflows,
                      cpu_roffsets, cpu_rdestinations, cpu_flow_idx,
                      Excess_total, mark, scanned);

        printf("After global relabel--------------------\n");
        // //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);

        // printf("Excess total : %d\n",*Excess_total);
        // printExcessFlow(V,cpu_excess_flow);

    }
    printf("Total kernel time: %.6f ms\n", totalMilliseconds);

}
