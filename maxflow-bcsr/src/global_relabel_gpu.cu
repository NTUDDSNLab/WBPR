#include "../include/parallel_graph.cuh"
#include "../include/utils.cuh"

void global_relabel_gpu(int V, int E, int source, int sink, 
                int *cpu_height, int *cpu_excess_flow, int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows,
                int *gpu_height, int *gpu_excess_flow, int *gpu_offsets, int *gpu_destinations, int* gpu_capacities, int* gpu_fflows,
                int *Excess_total, bool *mark, bool *scanned)
{
    for (int u = 0; u < V; u++) {
        // For all (u,v) belonging to E
        for (int i = cpu_offsets[u]; i < cpu_offsets[u + 1]; i++) {
            int v = cpu_destinations[i];

            if (cpu_height[u] > cpu_height[v] + 1) {
                
                // BUGS HERE! The cpu_excess_flow[u] might be smaller than cpu_fflow[i]
                // so we need to check if we can push more flow from u to v
                int flow;
                if (cpu_excess_flow[u] < cpu_fflows[i]) {
                    flow = cpu_excess_flow[u];
                } else {
                    flow = cpu_fflows[i];
                }

                cpu_excess_flow[u] -= flow;
                cpu_excess_flow[v] += flow;
                cpu_fflows[i] -= flow;
            }

        }
    }

    // Involve gpu_bfs
    printf("Invoking GPU BFS\n");

    // Initialize status array 

    // bool *gpu_mark;
    int *gpu_status;
    int *gpu_queue;
    int *gpu_level;
    int *gpu_Excess_total;
    int *gpu_queue_size;
    bool *terminate;

    // Allocate memory for status array and queue
    // CHECK(cudaMalloc(&gpu_mark, V * sizeof(bool)));
    CHECK(cudaMalloc(&gpu_status, V * sizeof(int)));
    CHECK(cudaMalloc(&gpu_queue, V * sizeof(int)));
    CHECK(cudaMalloc(&gpu_queue_size, sizeof(int)));
    CHECK(cudaMalloc(&gpu_level, sizeof(int)));
    CHECK(cudaMalloc(&terminate, sizeof(bool)));
    CHECK(cudaMalloc(&gpu_Excess_total, sizeof(int)));
    CHECK(cudaMemset(gpu_status, -1, V * sizeof(int)));
    CHECK(cudaMemset(gpu_queue, 0, V * sizeof(int)));
    CHECK(cudaMemset(gpu_level, 0, sizeof(int)));
    CHECK(cudaMemset(gpu_queue_size, 0, sizeof(int)));
    CHECK(cudaMemset(terminate, true, sizeof(bool)));

    printf("CPU Excess Total: %d\n", *Excess_total);


    // Copy data to GPU
    // CHECK(cudaMemcpy(gpu_mark, mark, V * sizeof(bool), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_height, cpu_height, V * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_excess_flow, cpu_excess_flow, V * sizeof(int), cudaMemcpyHostToDevice));
    // cudaMemcpy(gpu_offsets, cpu_offsets, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(gpu_destinations, cpu_destinations, E * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(gpu_capacities, cpu_capacities, E * sizeof(int), cudaMemcpyHostToDevice);
    CHECK(cudaMemcpy(gpu_fflows, cpu_fflows, E * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_Excess_total, Excess_total, sizeof(int), cudaMemcpyHostToDevice));


    // Configure the GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    dim3 num_blocks(deviceProp.multiProcessorCount * numBlocksPerSM);
    dim3 block_size(numThreadsPerBlock);

    void* kernel_args[] = {
        &V, &E, &source, &sink,
        &gpu_height, &gpu_excess_flow, &gpu_offsets, &gpu_destinations, &gpu_capacities, &gpu_fflows,
        &gpu_status, &gpu_queue, &gpu_queue_size, &gpu_level, &gpu_Excess_total, &terminate
    };


    // Invoke the kernel using cooperatvie groups
    cudaError_t cudaStatus;
    cudaStatus = cudaLaunchCooperativeKernel((void*)global_relabel_gpu_kernel, num_blocks, block_size, kernel_args);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error, for example, by cleaning up resources and exiting
        exit(1);
    }

    cudaDeviceSynchronize();

    // Copy data back to CPU
    CHECK(cudaMemcpy(cpu_height, gpu_height, V * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cpu_excess_flow, gpu_excess_flow, V * sizeof(int), cudaMemcpyDeviceToHost));
    // cudaMemcpy(cpu_offsets, gpu_offsets, (V + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(cpu_destinations, gpu_destinations, E * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(cpu_capacities, gpu_capacities, E * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK(cudaMemcpy(cpu_fflows, gpu_fflows, E * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Excess_total, gpu_Excess_total, sizeof(int), cudaMemcpyDeviceToHost));

    
    // Free memory
    CHECK(cudaFree(gpu_status));
    CHECK(cudaFree(gpu_queue));
    CHECK(cudaFree(gpu_level));
    CHECK(cudaFree(gpu_Excess_total));
    CHECK(cudaFree(gpu_queue_size));
    CHECK(cudaFree(terminate));
    
}

// Use Bottom-up BFS from sink to source
__global__ void global_relabel_gpu_kernel(int V, int E, int source, int sink,
                int *gpu_height, int *gpu_excess_flow, int *gpu_offsets, int *gpu_destinations, int* gpu_capacities, int* gpu_fflows,
                int *gpu_status, int *gpu_queue, int* gpu_queue_size, int *gpu_level, int *gpu_Excess_total, bool* terminate)
{

    grid_group grid = this_grid();
    unsigned int idx = (blockIdx.x*blockDim.x) + threadIdx.x;

    // Push the sink to the queue
    if (idx == 0) {
        gpu_status[sink] = 0;
        *gpu_queue_size = 0;
        *gpu_level = 1;
    }
    grid.sync();

    // Process the backward BFS
    while(true) {
        // Scan the gpu_status array and push the frontier to the queue
        for (int i = idx; i < V; i += blockDim.x * gridDim.x) {
            // Push the unvisited vertices to the queue
            if (gpu_status[i] == -1) {
                // Push i to the queue
                gpu_queue[atomicAdd(gpu_queue_size, 1)] = i;
            }
        }
        grid.sync();

        // Process the queue - bottom-up thread-centric version
        for (int i = idx; i < *gpu_queue_size; i += blockDim.x * gridDim.x) {
            int u = gpu_queue[i];
            for (int j = gpu_offsets[u]; j < gpu_offsets[u + 1]; j++) {
                int v = gpu_destinations[j];
                if (gpu_status[v] >= 0 && gpu_fflows[j] > 0) {
                    gpu_status[u] = *gpu_level + 1;
                    gpu_height[u] = gpu_status[u];
                    *terminate = false; // If some vertices are visited, then the BFS is not terminated
                    break;
                }
            }
        }
        grid.sync();

        // If the terminate flag is raise, break the loop
        if (*terminate) {
            break;
        }
        grid.sync();
        
        if (idx == 0) {
            *gpu_queue_size = 0;
            *gpu_level = *gpu_level + 1;
            *terminate = true;
        }
        grid.sync();
    }
    grid.sync();

    // Update the excess flow
    for (int i = idx; i < V; i += blockDim.x * gridDim.x) {
        if (gpu_status[i] == -1 && gpu_excess_flow[i] > 0 && i != source && i != sink) {
            // printf("Remove excess flow from %d (%d) \n", i, gpu_excess_flow[i]);
            // Use atomicSub to avoid race condition
            atomicSub(gpu_Excess_total, gpu_excess_flow[i]);
            gpu_excess_flow[i] = 0;
        }
    }
}
