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
        __syncthreads(); // FIXME: Find why sync is required here

        // cycle value is decreased
        cycle = cycle - 1;
    }
}