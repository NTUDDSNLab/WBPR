# Engineering A Workload-balanced Push-relabel Algorithms for Extremely Large Graphs on GPUs

## 1. Getting started Instructions.
- Clone this project
- Hardware:
    - `CPU x86_64` (Test on Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz)
    - `NVIDIA GPU (arch>=86)` with device memory >= 12GB.(Support NVIDIA RTX3080(sm_86)). Note that we mainly evaluate our experience on RTX3090. The execution time could be different with different devices.
- OS & Compler:
    - `Ubuntu 22.04`
    - `CUDA = 11.6`
    - `nvcc = 11.6` 
- Important Files/Directories
    - `maxflow-cuda/`: contains all source code of parallel push-relabel algorithm using bidirectional CSR (BCSR), including thread-centric and vertex-centric approach.
    - `maxflow-serial/`: contains all source code of serial version of push-relabel algorithm on CPU.
    - `network/`: contains the network generator.
        - `DIMACS`: the first DIMACS implementation challenge [source](http://archive.dimacs.rutgers.edu/pub/netflow/)
        - `GTgraph`: the modified version from [here](https://www.cse.psu.edu/~kxm85/software/GTgraph/)

        