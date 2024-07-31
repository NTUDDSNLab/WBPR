#include"../include/parallel_graph.cuh"

#ifdef WORKLOAD
__global__ void copyFromStaticToArray(unsigned long long* tempArray, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        tempArray[idx] = warpExecutionTime[idx];
    }
}
#endif // WORKLOAD



__global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx)
{
    // u'th node is operated on by the u'th thread
    grid_group grid = this_grid();
    unsigned int idx = (blockIdx.x*blockDim.x) + threadIdx.x;

    // cycle value is set to KERNEL_CYCLES as required 
    int cycle = (KERNEL_CYCLES);  

#ifdef WORKLOAD
    unsigned long long start, end;
    // Initialize the warpExecutionTime
    if (idx % 32 == 0) {
        warpExecutionTime[idx / 32] = 0;
    }
#endif // WORKLOAD

    while (cycle > 0) {

#ifdef WORKLOAD
        // Initialize the warpExecutionTime
        if (idx % 32 == 0) {
            start = clock64();
        }
#endif // WORKLOAD

        for (unsigned int u = idx; u < V; u += blockDim.x * gridDim.x) {
            
            int e_dash, h_dash, h_double_dash, v, v_dash, d;
            int v_index = -1; // The index of the edge of u to v_dash
            bool vinReverse = false;
            //printf("u: %d, excess_flow: %d, height: %d\n", u, gpu_excess_flow[u], gpu_height[u]);
            // Find the activate nodes
            if (gpu_excess_flow[u] > 0 && gpu_height[u] < V && u != source && u != sink) {
                
                e_dash = gpu_excess_flow[u];
                h_dash = INF;
                v_dash = -1; // Modify from NULL to -1

                // For all (u, v) belonging to E_f (residual graph edgelist)
                // Find (u, v) in both CSR format and revesred CSR format
                for (int i = gpu_offsets[u]; i < gpu_offsets[u + 1]; i++) {
                    v = gpu_destinations[i];
                    if (gpu_fflows[i] > 0) {
                        h_double_dash = gpu_height[v];
                        if (h_double_dash < h_dash) {
                            v_dash = v;
                            h_dash = h_double_dash;
                            v_index = i;
                            vinReverse = false;
                        }
                    }
                }
                // Find (u, v) in reversed CSR format
                for (int i = gpu_roffsets[u]; i < gpu_roffsets[u + 1]; i++) {
                    v = gpu_rdestinations[i];
                    int flow_idx = gpu_flow_idx[i];
                    
                    //if (u==2) printf("v: %d, gpu_height[%d]: %d, h_double_dash: %d\n", v, v, gpu_height[v], h_double_dash);

                    if (gpu_bflows[flow_idx] > 0) {
                        h_double_dash = gpu_height[v];
                        if (h_double_dash < h_dash) {
                            v_dash = v;
                            h_dash = h_double_dash;
                            v_index = flow_idx; // Find the bug here!!!
                            vinReverse = true;
                        }
                    }
                }

                /* Push operation */
                if (v_dash == -1) {
                    /* If there is no connected neighbors */
                    gpu_height[u] = V;
                    //printf("[NO_NEIGHBOR] u: %d\n", u);
                } else {
                    if (gpu_height[u] > h_dash) {
                        
                        /* Find the proper flow to push */

                        /* Find flow[(u,v_dash)] in CSR */
                        if (!vinReverse) {
                            if (e_dash > gpu_fflows[v_index]) {
                                d = gpu_fflows[v_index];
                            } else {
                                d = e_dash;
                            }

                            //printf("[PUSH] u: %d, v_dash: %d, d: %d\n", u, v_dash, d);


                            /* Push flow to residual graph */
                            atomicAdd(&gpu_bflows[v_index], d);
                            atomicSub(&gpu_fflows[v_index], d);

                            /* Update Excess Flow */
                            atomicAdd(&gpu_excess_flow[v_dash], d);
                            atomicSub(&gpu_excess_flow[u], d);
                            

                        } else {
                            /* Find rflow[(u,v)] in reversed CSR */
                            if (e_dash > gpu_bflows[v_index]) {
                                d = gpu_bflows[v_index];
                            } else {
                                d = e_dash;
                            }

                            //printf("[PUSH] u: %d, v_dash: %d, d: %d\n", u, v_dash, d);
                            /* Push flow to residual graph */

                            /* Push flow to residual graph */
                            atomicAdd(&gpu_fflows[v_index], d);
                            atomicSub(&gpu_bflows[v_index], d);

                            /* Update Excess Flow */
                            atomicAdd(&gpu_excess_flow[v_dash], d);
                            atomicSub(&gpu_excess_flow[u], d);

                        }
                    } else {
                        /* Relabel operation */
                        gpu_height[u] = h_dash + 1;
                        //printf("[RELABEL] u: %d, h_dash: %d\n", u, h_dash);
                    } 
                }
            }
        }

#ifdef WORKLOAD
        // Sum up the execution time of each warp
        if (idx % 32 == 0) {
            end = clock64();
            warpExecutionTime[idx / 32] += (end - start);
        }
#endif // WORKLOAD

        // cycle value is decreased
        cycle = cycle - 1;
        grid.sync();
    }
}

inline __device__ void
scan_active_vertices(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, int* avq)
{
    unsigned int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
    grid_group grid = this_grid();
    
    /* Initialize the avq_size */
    if(idx == 0){
        avq_size = 0;
    }
    grid.sync();

#ifdef WORKLOAD
    unsigned long long start, end;
    if (idx % 32 == 0) {
        start = clock64();
    }
#endif // WORKLOAD

    /* Stride scan the V set */
    for (int u = idx; u < V; u+= blockDim.x * gridDim.x) {
        if (gpu_excess_flow[u] > 0 && gpu_height[u] < V && u != source && u != sink) {
            avq[atomicAdd(&avq_size, 1)] = u;
        }
    }
#ifdef WORKLOAD
    if (idx % 32 == 0) {
        end = clock64();
        warpExecutionTime[idx / 32] += (end - start);
    }
#endif // WORKLOAD

}


template <unsigned int tileSize>
inline __device__ int
iterative_search_neighbor(cg::thread_block_tile<tileSize> tile, int pos, int *sheight, int *svid, int* svidx, bool *vinReverse, int *v_index, int  V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx, int* avq)
{
    unsigned int idx = tile.thread_rank(); // 0~31
    int tileID = threadIdx.x / tileSize;
    int u = avq[pos];
    int e_dash = gpu_excess_flow[u];
    int h_dash = INF;
    int v_dash = NULL;
    int h_double_dash, v;

    if (idx == 0) {

        // For all (u, v) belonging to E_f (residual graph edgelist)
        // Find (u, v) in both CSR format and revesred CSR format
        for (int i = gpu_offsets[u]; i < gpu_offsets[u + 1]; i++) {
            v = gpu_destinations[i];
            if (gpu_fflows[i] > 0) {
                h_double_dash = gpu_height[v];
                if (h_double_dash < h_dash) {
                    v_dash = v;
                    h_dash = h_double_dash;
                    *v_index = i;
                    *vinReverse = false;
                }
            }
        }
        // Find (u, v) in reversed CSR format
        for (int i = gpu_roffsets[u]; i < gpu_roffsets[u + 1]; i++) {
            v = gpu_rdestinations[i];
            int flow_idx = gpu_flow_idx[i];
            if (gpu_bflows[flow_idx] > 0) {
                h_double_dash = gpu_height[v];
                if (h_double_dash < h_dash) {
                    v_dash = v;
                    h_dash = h_double_dash;
                    *v_index = flow_idx; // Find the bug here!!!
                    *vinReverse = true;
                }
            }
        }
    }
    tile.sync();
    *v_index = __shfl_sync(0xffffffff, *v_index, 0, 32);
    v_dash = __shfl_sync(0xffffffff, v_dash, 0, 32);


    return v_dash;
}





// NOTICE: vinReverse should be set to false when finding in CSR
template <unsigned int tileSize>
inline __device__ int
tiled_search_neighbor(cg::thread_block_tile<tileSize> tile, int pos, int *sheight, int *svid, int* svidx, bool *vinReverse, int *v_index, int  V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx, int* avq)
{
    unsigned int idx = tile.thread_rank(); // 0~31
    int tileID = threadIdx.x / tileSize;
    int u = avq[pos];
    int degree = gpu_offsets[u + 1] - gpu_offsets[u];
    int rdegree = gpu_roffsets[u + 1] - gpu_roffsets[u];
    int num_iters = (int)ceilf((float)degree / (float)tileSize);
    int num_r_iters = (int)ceilf((float)rdegree / (float)tileSize);

    int minH = INF;
    int minV = -1;
    
    /* Initialize the shared memory */
    sheight[threadIdx.x] = INF;
    svid[threadIdx.x] = -1;
    svidx[threadIdx.x] = -2;
    tile.sync();

    /* The idx beyound the degree will not be barriered
        for (int i = idx; i < degree; i += tile.size())
    */
    /* Scan all the neighbors of u */
    for (int i = 0; i < num_iters; i++) {
        /* Fetch all the neighbor of u to the shared memory */
        /* Be carefull!, the neighbors of u are not all in E_f */
        int v_pos, v;
        if (i * tileSize + idx < degree) {
            v_pos = gpu_offsets[u] + i * tileSize + idx;
            v = gpu_destinations[v_pos];
            if ((gpu_fflows[v_pos] > 0) && (v != source)) {
                sheight[threadIdx.x] = gpu_height[v];
                svid[threadIdx.x] = v;
                svidx[threadIdx.x] = v_pos;
            } else {
                sheight[threadIdx.x] = INF;
                svid[threadIdx.x] = -1;
                svidx[threadIdx.x] = -1;
            }
        } else {
            sheight[threadIdx.x] = INF;
            svid[threadIdx.x] = -1;
            svidx[threadIdx.x] = -1;
        }
        tile.sync();

        // Parallel reduction to find min
        for (unsigned int s = tile.size()/2; s > 0; s >>= 1) {
            if (idx < s) {
                if ((sheight[threadIdx.x] > sheight[threadIdx.x + s])) {
                    sheight[threadIdx.x] = sheight[threadIdx.x + s];
                    svid[threadIdx.x] = svid[threadIdx.x + s];
                    svidx[threadIdx.x] = svidx[threadIdx.x + s];
                }
            }
            tile.sync();
        }

        tile.sync();


        /* Use delegated thread to update the minimum height a tile finding in an iteration */
        if (idx == 0) {
            if (minH > sheight[threadIdx.x]) { // The address of the first thread in the tile
                minH = sheight[threadIdx.x];
                minV = svid[threadIdx.x];
                *v_index = svidx[threadIdx.x];
                *vinReverse = false;
            }
        }
        tile.sync();  
        svid[threadIdx.x] = -1;
        sheight[threadIdx.x] = INF;
        tile.sync();
    }

    tile.sync();

    /* Scan all the neighbors of u in reversed CSR */
    for (int i = 0; i < num_r_iters; i++) {
        /* Fetch all the neighbor of u to the shared memory */
        int v_pos, v;
        if (i * tileSize + idx < rdegree) { 
            v_pos = gpu_flow_idx[gpu_roffsets[u] + i * tileSize + idx];
            v = gpu_rdestinations[gpu_roffsets[u] + i * tileSize + idx];
            if ((gpu_bflows[v_pos] > 0) && (v != source)) {
                sheight[threadIdx.x] = gpu_height[v];
                svid[threadIdx.x] = v;
                svidx[threadIdx.x] = v_pos;
            } else {
                sheight[threadIdx.x] = INF;
                svid[threadIdx.x] = -1;
                svidx[threadIdx.x] = -2;
            }
        } else {
            sheight[threadIdx.x] = INF;
            svid[threadIdx.x] = -1;
            svidx[threadIdx.x] = -2;
        }
        tile.sync();

        /* Parallel reduction to find min */
        for (unsigned int s = tile.size()/2; s > 0; s >>= 1) {
            if (idx < s) {
                if ((sheight[threadIdx.x] > sheight[threadIdx.x + s])) {
                    sheight[threadIdx.x] = sheight[threadIdx.x + s];
                    svid[threadIdx.x] = svid[threadIdx.x + s];
                    svidx[threadIdx.x] = svidx[threadIdx.x + s];
                }
            }
            tile.sync();
        }
        tile.sync();

        /* Use delegated thread to update the minimum height a tile finding in an iteration */
        if (idx == 0) {
            if (minH > sheight[threadIdx.x]) {
                minH = sheight[threadIdx.x];
                minV = svid[threadIdx.x];
                *v_index = svidx[threadIdx.x];
                *vinReverse = true;
            }
        }
        tile.sync();  
        svid[threadIdx.x] = -1;
        sheight[threadIdx.x] = INF;
        svidx[threadIdx.x] = -2;
        tile.sync();
    }
    tile.sync();
    svid[threadIdx.x] = -1;
    sheight[threadIdx.x] = INF;

    tile.sync();


    //printf("[%d]idx: %d, minV: %d, minH: %d\n", tileID, idx, minV, minH);
    //tile.sync();

    return minV;
}


__global__ void coop_push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx, 
                                    int *avq, int* gpu_cycle)
{
    grid_group grid = this_grid();
    cg::thread_block block = cg::this_thread_block();
    const int tileSize = TILE_SIZE;
    cg::thread_block_tile<tileSize> tile = cg::tiled_partition<tileSize>(block);
    int numTilesPerBlock = (blockDim.x + tileSize - 1)/ tileSize;
    int numTilesPerGrid = numTilesPerBlock * gridDim.x;
    int tileIdx = blockIdx.x * numTilesPerBlock + block.thread_rank() / tileSize;
    int idx = (blockIdx.x*blockDim.x) + threadIdx.x;

    int minV = -1;
    bool vinReverse = false;
    int v_index = -1;
    int cycle = V;

    /* Allocate shared array for parallel reduction of finding minimum height */
    extern __shared__ int SharedMemory[];
    int* sheight = SharedMemory; // sdate store the temporary height of the neighbor of u in each tile
    int* svid = (int*)&SharedMemory[blockDim.x]; // svid store the temporary vertex ID of the neighbor of u in each tile
    int* svidx = (int*)&svid[blockDim.x]; // svidx store the temporary index of the neighbor of u in each tile

    // Print the information in each thread
    // printf("[%d]: tileIdx: %d, tile.thread_rank(): %d, threadIdx.x: %d, numTilesPerBlock: %d, numTilesPerGrid: %d\n", idx, tileIdx, tile.thread_rank(), threadIdx.x, numTilesPerBlock, numTilesPerGrid);


    while (cycle > 0) {
        
        scan_active_vertices(V, source, sink, gpu_height, gpu_excess_flow, avq);

        // if (idx == 0) 
        //    printf("----- << avq_size: %d >>-------------\n", avq_size);

        grid.sync();

        /* Early break condition: if avq_size == 0, cannot because 
           we have to increase the height of these vertices to exclude active vertices */
        if (avq_size == 0) {
            break;
        }
        v_index = -1;

        grid.sync();

#ifdef WORKLOAD
        unsigned long long start, end;
        if (idx % 32 == 0) {
            start = clock64();
        }
#endif // WORKLOAD

        // Use a tile for an active vertex
        for (int i = tileIdx; i < avq_size; i += numTilesPerGrid) {
            int u = avq[i];

            minV = tiled_search_neighbor<tileSize>(tile, i, sheight, svid, svidx, &vinReverse, &v_index, V, source, sink, gpu_height, gpu_excess_flow, gpu_offsets, gpu_destinations, gpu_capacities, gpu_fflows, gpu_bflows, gpu_roffsets, gpu_rdestinations, gpu_flow_idx, avq); 
            // minV = iterative_search_neighbor<tileSize>(tile, i, sheight, svid, svidx, &vinReverse, &v_index, V, source, sink, gpu_height, gpu_excess_flow, gpu_offsets, gpu_destinations, gpu_capacities, gpu_fflows, gpu_bflows, gpu_roffsets, gpu_rdestinations, gpu_flow_idx, avq);
            // Each thread print its minV
            // printf("[%d] sync, u: %d, minV: %d\n", threadIdx.x, u, minV);
            tile.sync();
            
            
            /* Let delegated thread to push or relabel */
            if (tile.thread_rank() == 0) {
                // printf("[%d] threadID: %d, tileIdx: %d, u: %d, minV: %d, v_index: %d, vinReverse: %d\n", threadIdx.x, idx, tileIdx,  u, minV, v_index, vinReverse);
                /* Perform push or relabel operation */
                if (minV == -1) {
                    // u has no neighbor in E_f
                    gpu_height[u] = V; // Set the height of u to be the maximum
                    //printf("[%d-INF] threadID: %d, u: %d, minV: %d\n", tileIdx, u, minV);
                } else {
                    // u has neighbor in E_f
                    if (gpu_height[u] > gpu_height[minV]) {
                        /* Perform push operation */
                        int d;
                        
                        /* Find flow[(u,v_dash)] in CSR */
                        if (!vinReverse) {
                            if (gpu_excess_flow[u] > gpu_fflows[v_index]) {
                                d = gpu_fflows[v_index];
                            } else {
                                d = gpu_excess_flow[u];
                            }

                            //printf("[%d-PUSH] u: %d, minV: %d, d: %d\n", tileIdx, u, minV, d);
                            /* Push flow to residual graph */
                            atomicAdd(&gpu_bflows[v_index], d);
                            atomicSub(&gpu_fflows[v_index], d);

                            /* Update Excess Flow */
                            atomicAdd(&gpu_excess_flow[minV], d);
                            atomicSub(&gpu_excess_flow[u], d);
                            

                        } else {
                            /* Find rflow[(u,v)] in reversed CSR */
                            if (gpu_excess_flow[u] > gpu_bflows[v_index]) {
                                d = gpu_bflows[v_index];
                            } else {
                                d = gpu_excess_flow[u];
                            }
                            
                            //printf("[%d-PUSH] u: %d, minV: %d, d: %d\n", tileIdx, u, minV, d);
                            /* Push flow to residual graph */

                            /* Push flow to residual graph */
                            atomicAdd(&gpu_fflows[v_index], d);
                            atomicSub(&gpu_bflows[v_index], d);

                            /* Update Excess Flow */
                            atomicAdd(&gpu_excess_flow[minV], d);
                            atomicSub(&gpu_excess_flow[u], d);

                        }
                    } else {
                        /* Perform relabel operation */
                        gpu_height[u] = gpu_height[minV] + 1;
                        //printf("[%d-RELABEL] u: %d, minV: %d\n", tileIdx, u, minV);
                    }
                }
            }
            tile.sync();  
        }
        
#ifdef WORKLOAD
        if (idx % 32 == 0) {
            end = clock64();
            warpExecutionTime[idx / 32] += (end - start);
        }
#endif // WORKLOAD
        
        grid.sync();
        cycle = cycle - 1;
    }
}

__global__ void
coop_simple_kernel(int V, int source, int sink, int *gpu_offsets)
{
    unsigned int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
    if (idx == 0) {
        printf("gpu_offsets: \n");
        for (int i = 0; i < V; i++) {
            printf("%d ", gpu_offsets[i]);
        }
    }
}

