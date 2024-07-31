#pragma once
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip> // put_time
#include <iostream>
#include <mutex>
#include <thread>


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
