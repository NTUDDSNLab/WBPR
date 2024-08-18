#ifndef __PARALLEL__GRAPH__HEADER__CUDA__
#define __PARALLEL__GRAPH__HEADER__CUDA__

#include <cuda.h>
#include <cooperative_groups.h>
#include <bits/stdc++.h>
#include <vector>
#include <limits.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

// macros declared

#define number_of_nodes V
#define number_of_edges E
#define threads_per_block 256
#define numBlocksPerSM 1
#define numThreadsPerBlock 1024
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



#ifdef DEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif


// function prototypes for parallel implementation

void preflow(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
             int *offsets, int *destinations, int* capacities, int* forward_flows, int* backward_flows, int *Excess_total);
void push_relabel(int algo_type, int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows,
                int *Excess_total, 
                int *gpu_height, int *gpu_excess_flow, 
                int *gpu_offsets, int* gpu_destinations, int* gpu_capacities, int* gpu_fflows, int* gpu_bflows, 
                int* avq, int* gpu_cycle);
void global_relabel(int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows, 
                int *Excess_total, bool *mark, bool *scanned);
void readgraph(char* filename, int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx);
void print(int V,int *cpu_height, int *cpu_excess_flow, int *cpu_rflowmtx, int *cpu_adjmtx);

bool checkEnd(int V, int E, int source, int sink, int* cpu_excess_flow);

// prototype for the push relabel kernel

__global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows);

__global__ void coop_push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows, 
                                    int* avq, int* gpu_cycle);

__global__ void coop_simple_kernel(int V, int source, int sink, int *gpu_offsets);

/* Global variables */
__device__ unsigned int avq_size;

#endif