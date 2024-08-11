#include"../include/parallel_graph.cuh"
#include "../include/utils.cuh"

#ifdef TIME_BREAKDOWN
// The array to accumulate the maximum time of each type
unsigned long long acc_max[PROFILE_NUM] = {0};

/* Reset tb_duration */
void initTimeBreakdown(unsigned long long *tb_duration) {
    CUDA_SAFECALL(cudaMemset(tb_duration, 0, PROFILE_NUM * numThreadsPerBlock * numSM * sizeof(unsigned long long)));
}

/* Calculate time breakdown per iteration */
void calculateTimeBreakdownPerIteration(
    unsigned long long *tb_duration, unsigned long long *acc_max, int group_size) 
{
    for (int i = 0; i < PROFILE_NUM; i++) {
        unsigned long long total = 0;
        unsigned long long cur_max = 0;
        unsigned int num_active_warps = 0;

        for (int j = 0; j < numThreadsPerBlock * numBlocksPerSM * numSM; j+= group_size) {
            unsigned long long time = tb_duration[i * numThreadsPerBlock * numSM + j];
            if (time == 0) {
                continue;
            }
            num_active_warps++;
            total += time;
            if (time > cur_max) {
                cur_max = time;
            }
        }
        acc_max[i] += cur_max;
    }
}

void reportBreakdownData(float totalExeTime)
{
    /* Format */
    std::string pad(100, '-');
    std::string pad2(33, ' ');
    std::string pad3(100, '*');
    printf("%s\n", pad.c_str());
    printf("%s Time Breakdown %s\n", pad2.c_str(), pad2.c_str());
    printf("%s\n", pad.c_str());
    /* Get GPU frequency */
    int GPUfreqKHz;
    cudaError_t cudaStatus = cudaDeviceGetAttribute(&GPUfreqKHz, cudaDevAttrClockRate, 0);
    if (cudaStatus != cudaSuccess) {
        printf("Failed to get GPU clock rate: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
    unsigned int GPUfreqMHz = GPUfreqKHz / 1000;
    printf("GPU Operating Frequency: %u MHz\n", GPUfreqMHz);

    /* Caculate the portion between the maximum time of each time and the totalExeTime. */
    float portion[PROFILE_NUM + 1] = {0};
    float acc_portion = 0;
    for (int i = 0; i < PROFILE_NUM; i++) {
        portion[i] = (float)acc_max[i] / GPUfreqMHz / 1000 / (float)totalExeTime;
        acc_portion += portion[i];
    }
    portion[PROFILE_NUM] = 1 - acc_portion;
    /* Print the time breakdown */
    printf("Kernel execution time: %.6f ms\n", totalExeTime);
    printf("Neighbor searching time: %.3f ms(%.3f%%) \nBackward finding time: %.3f ms(%.3f%%) \nOther time: %.3f ms(%.3f%%)\n",
           portion[0] * totalExeTime, portion[0] * 100,
           portion[1] * totalExeTime, portion[1] * 100,
           portion[2] * totalExeTime, portion[2] * 100);

}

/* Report time breakdown, called at the end of the kernel */
void report_breakdown_data(unsigned long long *tb_duration, float totalExeTime) {

    /* Format */
    std::string pad(100, '-');
    std::string pad2(33, ' ');
    std::string pad3(100, '*');
    printf("%s\n", pad.c_str());
    printf("%s Time Breakdown %s\n", pad2.c_str(), pad2.c_str());
    printf("%s\n", pad.c_str());

    /* Find average, min, max executing time in duration[] of each types */
    for (int i = 0; i < PROFILE_NUM; i++) {
        unsigned long long total = 0;
        unsigned long long max = 0;
        unsigned long long min = tb_duration[i * numThreadsPerBlock * numSM];
        unsigned int num_active_warps = 0;
        /* Get GPU frequency */
        int GPUfreqKHz;
        cudaError_t cudaStatus = cudaDeviceGetAttribute(&GPUfreqKHz, cudaDevAttrClockRate, 0);
        if (cudaStatus != cudaSuccess) {
            printf("Failed to get GPU clock rate: %s\n", cudaGetErrorString(cudaStatus));
            exit(1);
        }
        unsigned int GPUfreqMHz = GPUfreqKHz / 1000;
        printf("GPU Operating Frequency: %u MHz\n", GPUfreqMHz);

        for (int j = 0; j < numThreadsPerBlock * numBlocksPerSM * numSM; j+= WARP_SIZE) {
            unsigned long long time = tb_duration[i * numThreadsPerBlock * numSM + j];
            if (time == 0) {
                // printf("!!!!!!!!!!! ZERO time: gw: %d, laneId: %d \n", (j/WARP_SIZE), (j%WARP_SIZE));
                continue;
            }
            num_active_warps++;
            total += time;
            if (time > max) {
                max = time;
            }
            if (time < min) {
                min = time;
            }
        }
        float average = 0;
        if (num_active_warps > 0) {
            average = total / (num_active_warps) / GPUfreqMHz / 1000;
        } 
        printf("Kernel execution time: %.6f ms\n", totalExeTime);
        printf("Average(type-%d) time: %.6f ms\n", i, average);
        printf("Max(type-%d) time: %.6f ms\n", i, (float)(max / GPUfreqMHz / 1000));
        printf("Min(type-%d) time: %.6f ms\n", i, (float)(min / GPUfreqMHz / 1000));
        printf("%s\n", pad3.c_str());
    }
}

#endif /* TIME_BREAKDOWN */




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
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows,
                int *Excess_total, 
                int *gpu_height, int *gpu_excess_flow, 
                int *gpu_offsets, int* gpu_destinations, int* gpu_capacities, int* gpu_fflows,
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
    // dim3 num_blocks(1);

    // dim3 num_blocks(deviceProp.multiProcessorCount * numBlocksPerSM / 16);
    dim3 num_blocks(numBlocksPerSM * numSM);
    dim3 block_size(numThreadsPerBlock);
    // dim3 num_blocks(1);
    // dim3 block_size(64);

    // Calculate the usage of shared memory
    size_t sharedMemSize = 3 * block_size.x * sizeof(int);

#ifdef WORKLOAD
    int max_iter = 1;
    int cur_iter = 0;
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

#ifdef TIME_BREAKDOWN

    /* Initialize */
    unsigned long long *tb_duration;
    CUDA_SAFECALL(cudaMallocManaged(&tb_duration, PROFILE_NUM * numThreadsPerBlock * numSM * sizeof(unsigned long long)));
    CUDA_SAFECALL(cudaMemset(tb_duration, 0, PROFILE_NUM * numThreadsPerBlock * numSM * sizeof(unsigned long long)));

#endif /* TIME_BREAKDOWN */

    // Print the configuration
    // Print GPU device name
    printf("GPU Device: %s\n", deviceProp.name);
    printf("Number of blocks: %d\n", num_blocks.x);
    printf("Number of threads per block: %d\n", block_size.x);
    printf("Total warps: %d\n", totalWarps);
    printf("Shared memory size: %lu\n", sharedMemSize);


#ifdef TIME_BREAKDOWN
    void* original_kernel_args[] = {&V, &source, &sink, &gpu_height, &gpu_excess_flow, 
                        &gpu_offsets, &gpu_destinations, &gpu_capacities, &gpu_fflows, &tb_duration};

    void* kernel_args[] = {&V, &source, &sink, &gpu_height, &gpu_excess_flow, 
                        &gpu_offsets, &gpu_destinations, &gpu_capacities, &gpu_fflows, 
                        &gpu_avq, &gpu_cycle, &tb_duration};
#else /* !TIME_BREAKDOWN */
    void* original_kernel_args[] = {&V, &source, &sink, &gpu_height, &gpu_excess_flow, 
                        &gpu_offsets, &gpu_destinations, &gpu_capacities, &gpu_fflows};

    void* kernel_args[] = {&V, &source, &sink, &gpu_height, &gpu_excess_flow, 
                        &gpu_offsets, &gpu_destinations, &gpu_capacities, &gpu_fflows, 
                        &gpu_avq, &gpu_cycle};
#endif /* !TIME_BREAKDOWN */
    


    // initialising mark values to false for all nodes
    for(int i = 0; i < V; i++)
    {
        mark[i] = false;
    }
    // for (int i = 0; i < 3; i++)
    int iter = 0;
    while((cpu_excess_flow[source] + cpu_excess_flow[sink]) < *Excess_total)
    {
        printf("cpu_excess_flow[source]: %d, cpu_excess_flow[sink]: %d\n",cpu_excess_flow[source], cpu_excess_flow[sink]);

        //printf("gpu_excess_flow[source]: %d, gpu_excess_flow[sink]: %d\n",gpu_excess_flow[source], gpu_excess_flow[sink]);
        // copying height values to CUDA device global memory
        CHECK(cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_excess_flow, cpu_excess_flow, V*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_fflows, cpu_fflows, E*sizeof(int), cudaMemcpyHostToDevice));
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
        
        CHECK(cudaDeviceSynchronize());
        timer.stop();
        totalMilliseconds += timer.elapsed();



        printf("Kernel invoked\n");

        // copying height, excess flow and residual flow values from device to host memory
        CHECK(cudaMemcpy(cpu_height,gpu_height,V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_excess_flow,gpu_excess_flow,V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_fflows,gpu_fflows, E*sizeof(int),cudaMemcpyDeviceToHost));

#ifdef WORKLOAD

        // Copy warp execution time from device to host
        copyFromStaticToArray<<<num_blocks, block_size>>>(gpu_warpExecutionTime, num_warps);
        cudaDeviceSynchronize();

        CHECK(cudaMemcpy(tempWarpExecution, gpu_warpExecutionTime, num_warps*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        if (cur_iter < max_iter) {
            for (int i = 0; i < num_warps; i++) {
                cpuWarpExecution[i] += tempWarpExecution[i];
            }
        }
        cur_iter++;
#endif // WORKLOAD

#ifdef TIME_BREAKDOWN
        /* Calculate time breakdown */
        if (algo_type == 0) {
            // Thread-centric approach
            calculateTimeBreakdownPerIteration(tb_duration, acc_max, 1);
        } else {
            // Vertex-centric approach
            calculateTimeBreakdownPerIteration(tb_duration, acc_max, WARP_SIZE);
        }

        // // Reset tb_duration
        initTimeBreakdown(tb_duration);

#endif /* TIME_BREAKDOWN */

        printf("Before global relabel--------------------\n");
        printf("Excess total: %d\n",*Excess_total);

        global_relabel_gpu(V, E, source, sink, 
                        cpu_height, cpu_excess_flow, cpu_offsets, cpu_destinations, cpu_capacities, cpu_fflows,
                        gpu_height, gpu_excess_flow, gpu_offsets, gpu_destinations, gpu_capacities, gpu_fflows,
                        Excess_total, mark, scanned);

        printf("After global relabel--------------------\n");
        // //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);
        iter++;

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
    
#ifdef TIME_BREAKDOWN
    // launch kernel to print device scanTime and backwardTime
    // printDeviceTime<<<num_blocks, block_size>>>();
    // cudaDeviceSynchronize();
    
    // report_breakdown_data(tb_duration, totalMilliseconds);
    // FinializeTimeBreakdown();
    reportBreakdownData(totalMilliseconds);
    cudaFree(tb_duration);
#endif /* TIME_BREAKDOWN */


}
