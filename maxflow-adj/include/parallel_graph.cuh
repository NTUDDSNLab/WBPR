#ifndef __PARALLEL__GRAPH__HEADER__CUDA__
#define __PARALLEL__GRAPH__HEADER__CUDA__

#include<cuda.h>
#include<bits/stdc++.h>
#include<string.h>
#include<limits.h>
#include "../include/utils.cuh"


// macros declared

#define number_of_nodes *V
#define number_of_edges *E
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF INT_MAX
#define IDX(x,y) ( ( (x)*(number_of_nodes) ) + (y) )
#define KERNEL_CYCLES V
#define ull unsigned long long


// function prototypes for parallel implementation

void preflow(ull *V, ull source, ull sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total);
void push_relabel(ull *V, ull *gpu_V, ull source, ull sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx, int *gpu_rflowmtx);
void global_relabel(ull *V, ull source, ull sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, bool *mark, bool *scanned);
void print(ull *V,int *cpu_height, int *cpu_excess_flow, int *cpu_rflowmtx, int *cpu_adjmtx);
// function to read different types of input graphs
void readFromDIMACSFormat(std::string filename, ull *V, ull *E, ull *source, ull *sink, int **cpu_height, int **cpu_excess_flow, int **cpu_adjmtx, int **cpu_rflowmtx);
// void readFromSNAPFormat(std::string filename, ull *V, ull *E, ull *source, ull *sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx);

// prototype for the push relabel kernel

__global__ void push_relabel_kernel(ull *V, ull source, ull sink, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx,int *gpu_rflowmtx);

#endif