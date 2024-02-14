#include"../include/parallel_graph.cuh"


__global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx)
{
    // u'th node is operated on by the u'th thread
    unsigned int idx = (blockIdx.x*blockDim.x) + threadIdx.x;

    // cycle value is set to KERNEL_CYCLES as required 
    int cycle = (KERNEL_CYCLES);  

    while (cycle > 0) {

        for (unsigned int u = idx; u < V; u += blockDim.x * gridDim.x) {
            
            int e_dash, h_dash, h_double_dash, v, v_dash, d;
            int v_index = -1; // The index of the edge of u to v_dash
            bool vinReverse = false;

            // Find the activate nodes
            if (gpu_excess_flow[u] > 0 && gpu_height[u] < V && u != source && u != sink) {
                
                e_dash = gpu_excess_flow[u];
                h_dash = INF;
                v_dash = NULL;

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
                            v_index = flow_idx; // Find the bug here!!!
                            vinReverse = true;
                        }
                    }
                }

                /* Push operation */
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
            


            // if (threadIdx.x == 0)  printf("u : %d\n",u);

            /* Variables declared to be used inside the kernel :
            * e_dash - initial excess flow of node u
            * h_dash - height of lowest neighbor of node u
            * h_double_dash - used to iterate among height values to find h_dash
            * v - used to iterate among nodes to find v_dash
            * v_dash - lowest neighbor of node u 
            * d - flow to be pushed from node u
            */

            // int e_dash,h_dash,h_double_dash,v,v_dash,d;

            // if( (gpu_excess_flow[u] > 0) && (gpu_height[u] < V) )
            // {
            //     e_dash = gpu_excess_flow[u];
            //     h_dash = INF;
            //     v_dash = NULL;

            //     for(v = 0; v < V; v++)
            //     {
            //         // for all (u,v) belonging to E_f (residual graph edgelist)
            //         if(gpu_rflowmtx[IDX(u,v)] > 0)
            //         {
            //             h_double_dash = gpu_height[v];
            //             // finding lowest neighbor of node u
            //             if(h_double_dash < h_dash)
            //             {
            //                 v_dash = v;
            //                 h_dash = h_double_dash;
            //             }
            //         }
            //     }

            //     if(gpu_height[u] > h_dash)
            //     {
            //         /* height of u > height of lowest neighbor
            //         * Push operation can be performed from node u to lowest neighbor
            //         * All addition, subtraction and minimum operations are done using Atomics
            //         * This is to avoid anomalies in conflicts between multiple threads
            //         */

            //         // d captures flow to be pushed 
            //         d = e_dash;
            //         //atomicMin(&d,gpu_rflowmtx[IDX(u,v_dash)]);
            //         if(e_dash > gpu_rflowmtx[IDX(u,v_dash)])
            //         {
            //             d = gpu_rflowmtx[IDX(u,v_dash)];
            //         }
            //         // Residual flow towards lowest neighbor from node u is increased
            //         atomicAdd(&gpu_rflowmtx[IDX(v_dash,u)],d);

            //         // Residual flow towards node u from lowest neighbor is decreased
            //         atomicSub(&gpu_rflowmtx[IDX(u,v_dash)],d);

            //         // Excess flow of lowest neighbor and node u are updated
            //         atomicAdd(&gpu_excess_flow[v_dash],d);
            //         atomicSub(&gpu_excess_flow[u],d);
            //     }

            //     else
            //     {
            //         /* height of u <= height of lowest neighbor,
            //         * No neighbor with lesser height exists
            //         * Push cannot be performed to any neighbor
            //         * Hence, relabel operation is performed
            //         */

            //         gpu_height[u] = h_dash + 1;
            //     }

            // }

        }
        __syncthreads();

        // cycle value is decreased
        cycle = cycle - 1;
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
    
    /* Stride scan the V set */
    for (int u = idx; u < V; u+= blockDim.x * gridDim.x) {
        if (gpu_excess_flow[u] > 0 && gpu_height[u] < V && u != source && u != sink) {
            avq[atomicAdd(&avq_size, 1)] = u;
        }
    }
}



template <unsigned int tileSize>
inline __device__ int
tiled_search_neighbor(cg::thread_block_tile<tileSize> tile, int pos, int *sheight, int *svid, bool *vinReverse, int *v_index, int  V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx, int* avq)
{
    unsigned int idx = tile.thread_rank();
    int u = avq[pos];
    int degree = gpu_offsets[u + 1] - gpu_offsets[u];
    int rdegree = gpu_roffsets[u + 1] - gpu_roffsets[u];

    int minH = INF;
    int minV = -1;

    /* Scan all the neighbors of u */
    for (int i = idx; i < degree; i += tile.size()) {
        /* Fetch all the neighbor of u to the shared memory */
        /* Be carefull!, the neighbors of u are not all in E_f */
        int v_pos = gpu_offsets[u] + i;
        int v = gpu_destinations[v_pos];

        // Put only the neighbors of u that are in E_f
        if (gpu_fflows[v_pos] > 0) {
            sheight[idx] = gpu_height[v];
            svid[idx] = v;
        } else {
            sheight[idx] = INF;
            svid[idx] = -1;
        }

        tile.sync();

        /* FIXME: Parallel reduction to find min */
        for (int s = 1; s < tile.size(); s *= 2) {
            if (idx % (2*s) == 0) {
                if (sheight[idx] > sheight[idx + s] || svid[idx] == source) {
                    sheight[idx] = sheight[idx + s];
                    svid[idx] = svid[idx + s];
                } else {
                    sheight[idx] = sheight[idx];
                    svid[idx] = svid[idx];
                }
            }
            tile.sync();
        }
        tile.sync();

        /* Use delegated thread to update the minimum height a tile finding in an iteration */
        if (idx == 0) {
            if (minH > sheight[0]) {
                minH = sheight[0];
                minV = svid[0];
            }
        }
        tile.sync();  
    }
    tile.sync();

    /* Scan all the neighbors of u in reversed CSR */
    for (int i = idx; i < rdegree; i += tile.size()) {
        /* Fetch all the neighbor of u to the shared memory */
        int v_pos = gpu_flow_idx[gpu_roffsets[u] + i];
        int v = gpu_rdestinations[gpu_roffsets[u] + i];

        if (gpu_bflows[v_pos] > 0) {
            sheight[idx] = gpu_height[v];
            svid[idx] = v;
        } else {
            sheight[idx] = INF;
            svid[idx] = -1;
        }
        tile.sync();

        /* Parallel reduction to find min */
        for (int s = 1; s < tile.size(); s *= 2) {
            if (idx % (2*s) == 0) {
                if (sheight[idx] > sheight[idx + s] || svid[idx] == source) {
                    sheight[idx] = sheight[idx + s];
                    svid[idx] = svid[idx + s];
                } else {
                    sheight[idx] = sheight[idx];
                    svid[idx] = svid[idx];
                }
            }
            tile.sync();
        }
        tile.sync();

        /* Use delegated thread to update the minimum height a tile finding in an iteration */
        if (idx == 0) {
            if (minH > sheight[0]) {
                minH = sheight[0];
                minV = svid[0];
                *vinReverse = true;
            }
        }
        tile.sync();  
    }

    /* Parallel find the v_index of the minV */
    if (*vinReverse) {
        for (int i = idx; i < rdegree; i += tile.size()) {
            if (minV == gpu_rdestinations[gpu_roffsets[u] + i]) {
                *v_index = gpu_flow_idx[gpu_roffsets[u] + i];
            }
        }
    } else {
        for (int i = idx; i < degree; i += tile.size()) {
            if (minV == gpu_destinations[gpu_offsets[u] + i]) {
                *v_index = gpu_offsets[u] + i;
            }
        }
    }
    tile.sync();

    return minV;

}


__global__ void coop_push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx, 
                                    int *avq, int* gpu_cycle)
{
    grid_group grid = this_grid();
    cg::thread_block block = cg::this_thread_block();
    const int tileSize = 32;
    cg::thread_block_tile<tileSize> tile = cg::tiled_partition<tileSize>(block);
    int numTilesPerBlock = (blockDim.x + tileSize - 1)/ tileSize;
    int numTilesPerGrid = numTilesPerBlock * gridDim.x;
    int tileIdx = blockIdx.x * numTilesPerBlock + block.thread_rank() / tileSize;
    int idx = (blockIdx.x*blockDim.x) + threadIdx.x;

    int minV = -1;
    bool vinReverse = false;
    int v_index;

    /* Allocate shared array for parallel reduction of finding minimum height */
    extern __shared__ int SharedMemory[];
    int* sheight = SharedMemory; // sdate store the temporary height of the neighbor of u in each tile
    int* svid = (int*)&SharedMemory[blockDim.x]; // svid store the temporary vertex ID of the neighbor of u in each tile
    
    // Initialize the cycle value
    if (idx == 0) {
        *gpu_cycle = V;
    }
    grid.sync();
    
    while (*gpu_cycle > 0) {

        scan_active_vertices(V, source, sink, gpu_height, gpu_excess_flow, avq);
        grid.sync();

        /* Early break condition: if avq_size == 0, cannot because 
           we have to increase the height of these vertices to exclude active vertices */

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (avq_size == 0) {
               *gpu_cycle = 0;
            }
            printf("avq_size: %d, gpu_cycle:%d, V: %d\n", avq_size, *gpu_cycle, V);
        }

        grid.sync();

        // Use a tile for an active vertex
        for (int i = tileIdx; i < avq_size; i += numTilesPerGrid) {
            minV = tiled_search_neighbor<tileSize>(tile, i, sheight, svid, &vinReverse, &v_index, V, source, sink, gpu_height, gpu_excess_flow, gpu_offsets, gpu_destinations, gpu_capacities, gpu_fflows, gpu_bflows, gpu_roffsets, gpu_rdestinations, gpu_flow_idx, avq);
            tile.sync();
            
            int u = avq[i];
            
            /* Let delegated thread to push or relabel */
            if (minV != -1 && tile.thread_rank() == 0) {
                /* Perform push or relabel operation */
                printf("Idx: %d, u: %d, height[u]: %d, excess_flow[u]: %d, minV: %d, minH: %d\n", idx, u, gpu_height[u], gpu_excess_flow[u], minV,  gpu_height[minV]);
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

                        //printf("[PUSH] u: %d, v_dash: %d, d: %d\n", u, v_dash, d);
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

                        //printf("[PUSH] u: %d, v_dash: %d, d: %d\n", u, v_dash, d);
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
                }
            }
            tile.sync();      
        }
        grid.sync();
        if ((blockIdx.x*blockDim.x) + threadIdx.x == 0) {
            *gpu_cycle = *gpu_cycle - 1;
        }
        grid.sync();
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

