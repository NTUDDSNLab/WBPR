#pragma once
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip> // put_time
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>


#ifdef DEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif



#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__,                    \
              cudaGetErrorName(err), cudaGetErrorString(err));                 \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

#define DUMP(x)                                                                \
  do {                                                                         \
    std::cout << #x << ": " << x << std::endl;                                 \
  } while (0);

class CudaTimer {
public:
  CudaTimer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
  }

  ~CudaTimer() {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
  }

  void start() { cudaEventRecord(startEvent, 0); }

  void stop() {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
  }

  float elapsed() const {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    return milliseconds;
  }

private:
  cudaEvent_t startEvent, stopEvent;
};


#undef CEILING
#define CEILING(x, y) (((x) + (y)-1) / (y))

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            _exit(EXIT_FAILURE);                                            \
        }                                                                   \
    }

/*********************************************************************
 *
 *                   Device level utility functions
 *
 **********************************************************************/

// Get the SM id
__device__ __forceinline__ unsigned int get_smid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

// Get the warp id within the application
__device__ __forceinline__ unsigned int get_warpid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

// Get the line id within the warp
__device__ __forceinline__ unsigned int get_laneid(void) {
    unsigned int laneid;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid));
    return laneid;
}

// Get a global warp id
__device__ __forceinline__ int get_global_warp_id() {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    int l_thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;

    int l_warp_id = l_thread_id / 32;

    int n_warps = CEILING(blockDim.x * blockDim.y * blockDim.z, 32);

    int g_warp_id = block_id * n_warps + l_warp_id;

    return g_warp_id;
}

// Get a thread's CTA ID
__device__ __forceinline__ int4 get_ctaid(void) {
    int4 ret;
    asm("mov.u32 %0, %ctaid.x;" : "=r"(ret.x));
    asm("mov.u32 %0, %ctaid.y;" : "=r"(ret.y));
    asm("mov.u32 %0, %ctaid.z;" : "=r"(ret.z));
    return ret;
}

//  Get the number of CTA ids per grid
__device__ __forceinline__ int4 get_nctaid(void) {
    int4 ret;
    asm("mov.u32 %0, %nctaid.x;" : "=r"(ret.x));
    asm("mov.u32 %0, %nctaid.y;" : "=r"(ret.y));
    asm("mov.u32 %0, %nctaid.z;" : "=r"(ret.z));
    return ret;
}

// Device level sleep function
__device__ __forceinline__ void csleep(uint64_t clock_count) {
    if (clock_count == 0) return;
    clock_t start_clock = clock64();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count) {
        clock_offset = clock64() - start_clock;
    }
}

class Managed {
  public:
    void *operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        return ptr;
    }

    // void Managed::operator delete(void *ptr)
    void operator delete(void *ptr) { cudaFree(ptr); }

    void *operator new[](size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        return ptr;
    }
    // void Managed::operator delete[] (void* ptr) {
    void operator delete[](void *ptr) { cudaFree(ptr); }
};