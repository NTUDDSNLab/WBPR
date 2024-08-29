#ifndef __PARALLEL__GRAPH__HEADER__CUDA__
#define __PARALLEL__GRAPH__HEADER__CUDA__

#include <cuda.h>
#include <cooperative_groups.h>
#include <bits/stdc++.h>
#include <vector>
#include <limits.h>
#include "utils.cuh"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

// macros declared

#define number_of_nodes V
#define number_of_edges E
#define numBlocksPerSM 1
#define numSM 128
#define numThreadsPerBlock 1024
#define WARP_SIZE 32
#define totalThreads (numThreadsPerBlock * numBlocksPerSM * numSM)
#define numWarpsPerBlock (numThreadsPerBlock / 32)
#define totalWarps (numWarpsPerBlock * numBlocksPerSM * numSM)
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF INT_MAX
#define IDX(x,y) ( ( (x)*(number_of_nodes) ) + (y) )
#define KERNEL_CYCLES V
#define TILE_SIZE 32

#ifdef WORKLOAD
#define enoughArraySize 1000000
__device__ unsigned long long warpExecutionTime[enoughArraySize] = {0}; // Enough space for all warps in RTX 3090
__global__ void copyFromStaticToArray(unsigned long long* tempArray, int N);
#endif /* WORKLOAD */


#ifdef TIME_BREAKDOWN

/* Record outgoing edge scanning and backward edge searching */
#define PROFILE_NUM 2

__device__ unsigned long long ANNOTATE_START();
__device__ unsigned long long ANNOTATE_END(int types, unsigned long long start);



/* Annotation Macro : type: scan or backward */
// FIXME: Race condition in the following macros
// #define ANNOTATE_START(type) \
//     unsigned long long start_##type; \
//     if (threadIdx.x % WARP_SIZE == 0) { \
//         asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_##type)); \
//     } \
//     __syncwarp();

// #define ANNOTATE_END(type) \
//     unsigned long long end_##type; \
//     unsigned long long time_##type = 0; \
//     unsigned int warp_id_##type = threadIdx.x / WARP_SIZE + (blockIdx.x * numWarpsPerBlock); \
//     if (threadIdx.x % WARP_SIZE == 0) { \
//         asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_##type)); \
//         time_##type = end_##type - start_##type; \
//         type##Time[warp_id_##type] += time_##type; \
//     } \
//     __syncwarp();

// void InitializeTimeBreakdown();

// inline __device__ void InitializeTimeBreakdownDevice();

// __global__ void printDeviceTime();

// void FinializeTimeBreakdown();

// __global__ void copyScanToHost(unsigned long long* des,int N);

// __global__ void copyBackwardToHost(unsigned long long* des, int N);


void report_breakdown_data(float totalExeTime);
#endif /* TIME_BREAKDOWN */


#ifdef DEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif


// function prototypes for parallel implementation

void preflow(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
             int *offsets, int *destinations, int* capacities, int* forward_flows, int* backward_flows,
             int *roffsets, int* rdestinations, int* flow_idx, int *Excess_total);
void push_relabel(int algo_type, int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows,
                int* cpu_roffsets, int* cpu_rdestinations, int* cpu_flow_idx,
                int *Excess_total, 
                int *gpu_height, int *gpu_excess_flow, 
                int *gpu_offsets, int* gpu_destinations, int* gpu_capacities, int* gpu_fflows, int* gpu_bflows,
                int* gpu_roffsets, int* gpu_rdestinations, int* gpu_flow_idx, 
                int* avq, int* gpu_cycle);
void global_relabel(int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows, 
                int* cpu_roffsets, int* cpu_rdestinations, int* cpu_flow_idx,
                int *Excess_total, bool *mark, bool *scanned);
void readgraph(char* filename, int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx);
void print(int V,int *cpu_height, int *cpu_excess_flow, int *cpu_rflowmtx, int *cpu_adjmtx);

bool checkEnd(int V, int E, int source, int sink, int* cpu_excess_flow);

// prototype for the push relabel kernel


#ifdef TIME_BREAKDOWN
__global__ void coop_push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx, 
                                    int *avq, int* gpu_cycle, unsigned long long *tb_duration);

__global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx, unsigned long long *tb_duration);
#else /* !TIME_BREAKDOWN */
__global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx);

__global__ void coop_push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx, 
                                    int* avq, int* gpu_cycle);
#endif /* TIME_BREAKDOWN */
__global__ void coop_simple_kernel(int V, int source, int sink, int *gpu_offsets);

/* Global variables */
__device__ unsigned int avq_size;

#endif