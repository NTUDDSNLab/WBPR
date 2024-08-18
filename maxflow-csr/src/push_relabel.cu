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


void push_relabel(int algo_type, int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows,
                int *Excess_total, 
                int *gpu_height, int *gpu_excess_flow, 
                int *gpu_offsets, int* gpu_destinations, int* gpu_capacities, int* gpu_fflows, int* gpu_bflows, 
                int* gpu_avq, int* gpu_cycle)
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
    printf("Inside push_relabel\n");


    // Configure the GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    dim3 num_blocks(deviceProp.multiProcessorCount * numBlocksPerSM);
    dim3 block_size(numThreadsPerBlock);
    // dim3 num_blocks(1);
    // dim3 block_size(64);

    // Calculate the usage of shared memory
    size_t sharedMemSize = 3 * block_size.x * sizeof(int);

#ifdef WORKLOAD
    // Caculate the total number of warps
    int num_warps = (block_size.x * num_blocks.x) / 32;
    
    // Allocate device buffer for warp execution time
    unsigned long long *gpu_warpExecutionTime;
    CHECK(cudaMalloc((void**)&gpu_warpExecutionTime, num_warps*sizeof(unsigned long long)));

    // Allocate host buffer for warp execution time
    unsigned long long *cpuWarpExecution = (unsigned long long*)malloc(num_warps*sizeof(unsigned long long));
    unsigned long long *tempWarpExecution = (unsigned long long*)malloc(num_warps*sizeof(unsigned long long));
    for (int i = 0; i < num_warps; i++) {
        cpuWarpExecution[i] = 0;
    }
#endif // WORKLOAD

    // Print the configuration
    // Print GPU device name
    printf("GPU Device: %s\n", deviceProp.name);
    printf("Number of blocks: %d\n", num_blocks.x);
    printf("Number of threads per block: %d\n", block_size.x);
    printf("Shared memory size: %lu\n", sharedMemSize);

    void* original_kernel_args[] = {&V, &source, &sink, &gpu_height, &gpu_excess_flow, 
                        &gpu_offsets, &gpu_destinations, &gpu_capacities, &gpu_fflows, &gpu_bflows, 
                        };


    void* kernel_args[] = {&V, &source, &sink, &gpu_height, &gpu_excess_flow, 
                        &gpu_offsets, &gpu_destinations, &gpu_capacities, &gpu_fflows, &gpu_bflows,  
                        &gpu_avq, &gpu_cycle};


    // initialising mark values to false for all nodes
    for(int i = 0; i < V; i++)
    {
        mark[i] = false;
    }
    //for (int i = 0; i < 3; i++)
    while((cpu_excess_flow[source] + cpu_excess_flow[sink]) < *Excess_total)
    {
        printf("cpu_excess_flow[source]: %d, cpu_excess_flow[sink]: %d\n",cpu_excess_flow[source], cpu_excess_flow[sink]);

        //printf("gpu_excess_flow[source]: %d, gpu_excess_flow[sink]: %d\n",gpu_excess_flow[source], gpu_excess_flow[sink]);
        // copying height values to CUDA device global memory
        CHECK(cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_excess_flow, cpu_excess_flow, V*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_fflows, cpu_fflows, E*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_bflows, cpu_bflows, E*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(gpu_cycle, V, sizeof(int))); // Reset the gpu_cycle to V


        printf("Invoking kernel\n");

        cudaError_t cudaStatus;
        timer.start();
        if (algo_type == 0) {
            // Thread-centric approach
            cudaStatus = cudaLaunchCooperativeKernel((void*)push_relabel_kernel, num_blocks, block_size, original_kernel_args, sharedMemSize, 0);
        } else {
            // Vertex-centric approach
            cudaStatus = cudaLaunchCooperativeKernel((void*)coop_push_relabel_kernel, num_blocks, block_size, kernel_args, sharedMemSize, 0);
        }
        
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(cudaStatus));
            // Handle the error, for example, by cleaning up resources and exiting
            exit(1);
        }
        
        cudaDeviceSynchronize();
        timer.stop();
        totalMilliseconds += timer.elapsed();



        printf("Kernel invoked\n");

        // copying height, excess flow and residual flow values from device to host memory
        CHECK(cudaMemcpy(cpu_height,gpu_height,V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_excess_flow,gpu_excess_flow,V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_fflows,gpu_fflows, E*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_bflows,gpu_bflows, E*sizeof(int),cudaMemcpyDeviceToHost));

#ifdef WORKLOAD

        // Copy warp execution time from device to host
        copyFromStaticToArray<<<num_blocks, block_size>>>(gpu_warpExecutionTime, num_warps);
        cudaDeviceSynchronize();

        CHECK(cudaMemcpy(tempWarpExecution, gpu_warpExecutionTime, num_warps*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_warps; i++) {
            cpuWarpExecution[i] += tempWarpExecution[i];
        }
#endif // WORKLOAD



        // printf("Before global relabel--------------------\n");


        //printExcessFlow(V,cpu_excess_flow);
        //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        //printf("Excess total : %d\n",*Excess_total);
        // perform the global_relabel routine on host
        // printf("Before global relabel, Excess total : %d\n",*Excess_total);

        global_relabel(V, E, source,sink,cpu_height,cpu_excess_flow,
                      cpu_offsets,cpu_destinations, cpu_capacities, cpu_fflows, cpu_bflows,
                      Excess_total, mark, scanned);

        printf("After global relabel--------------------\n");
        // //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);
        
        // printf("Excess total : %d\n",*Excess_total);
        // printExcessFlow(V,cpu_excess_flow);

    }
    printf("Total kernel time: %.6f ms\n", totalMilliseconds);

#ifdef WORKLOAD
    printf("------------<< Workload Information >>------------\n");
    printf("#warps: %d\n", num_warps);
    printf("Warp execution time:\n");
    for (int i = 0; i < num_warps; i++) {
        printf("%llu ", cpuWarpExecution[i]);
    }
    printf("\n");

    // Free device buffer for warp execution time
    CHECK(cudaFree(gpu_warpExecutionTime));

    // Free host buffer for warp execution time
    free(cpuWarpExecution);
    free(tempWarpExecution);

#endif // WORKLOAD


}
