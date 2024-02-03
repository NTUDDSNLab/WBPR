#ifndef __PARALLEL__GRAPH__HEADER__CUDA__
#define __PARALLEL__GRAPH__HEADER__CUDA__

#include <cuda.h>
#include <bits/stdc++.h>
#include <vector>
#include <limits.h>
#include "utils.cuh"

// macros declared

#define number_of_nodes V
#define number_of_edges E
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF INT_MAX
#define IDX(x,y) ( ( (x)*(number_of_nodes) ) + (y) )
#define KERNEL_CYCLES V

// function prototypes for parallel implementation

void preflow(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
             int *offsets, int *destinations, int* capacities, int* forward_flows, int* backward_flows,
             int *roffsets, int* rdestinations, int* flow_idx, int *Excess_total);
void push_relabel(int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows,
                int* cpu_roffsets, int* cpu_rdestinations, int* cpu_flow_idx,
                int *Excess_total, 
                int *gpu_height, int *gpu_excess_flow, 
                int *gpu_offsets, int* gpu_destinations, int* gpu_capacities, int* gpu_fflows, int* gpu_bflows,
                int* gpu_roffsets, int* gpu_rdestinations, int* gpu_flow_idx);
void global_relabel(int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows, 
                int* cpu_roffsets, int* cpu_rdestinations, int* cpu_flow_idx,
                int *Excess_total, bool *mark, bool *scanned);
void readgraph(char* filename, int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx);
void print(int V,int *cpu_height, int *cpu_excess_flow, int *cpu_rflowmtx, int *cpu_adjmtx);

bool checkEnd(int V, int E, int source, int sink, int* cpu_excess_flow);

// prototype for the push relabel kernel

__global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, 
                                    int *gpu_offsets,int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                    int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx);

#endif