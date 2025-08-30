# Accelerating Push-Relabel Algorithm on GPU via Two-Level Parallelism Paradigm and Efficient CSR Designs

## 1. Getting started Instructions.
- Clone this project
- Hardware:
    - `CPU x86_64` (Test on Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz)
    - `NVIDIA GPU (arch>=86)` with device memory >= 12GB.(Support NVIDIA RTX3080(sm_86)). Note that we mainly evaluate our experience on RTX3090. The execution time could be different with different devices.
- OS & Compler:
    - `Ubuntu 22.04`
    - `CUDA >= 11.6`
    - `nvcc >= 11.6` 
- Important Files/Directories
    - `maxflow-cuda/`: contains all source code of parallel push-relabel algorithm using reversed CSR (RCSR), including thread-centric(TC) and vertex-centric(VC) approach.
    - `maxflow-bcsr`: contain TC, VC push-relabel algorithms upon bidirectional CSR (BCSR). Backward edges of a vertex are continuously appended to the end of the its forward edges.
    - `(TODO) maxflow-bcsr2`: the optimized version of BCSR, where the backward edge is placed right after a forward edges, namely, flow[2i] stores the forward edge value, and flow[2i+1] stores the corresponding backward edge.
    - `maxflow-serial/`: contains all source code of serial version of push-relabel algorithm on CPU.
    - `network/`: contains the network generator.
        - `DIMACS`: the first DIMACS implementation challenge [source](http://archive.dimacs.rutgers.edu/pub/netflow/)
        - `GTgraph`: the modified version from [here](https://www.cse.psu.edu/~kxm85/software/GTgraph/)

## 2. Setup & Experiments

### (0) Choose the right branch
Please choose the underlying CSR implementation
```
git checkout <branch_name>
```
branch_name:
* `csr`: the original CSR implementation
* `rcsr`: the reversed CSR implementation
* `bcsr`: the bidirectional CSR implementation

### (1) Configure the Makefile
Please uncomment the compiling flag of WORKLOAD in both CC_FLAGS and NVCC_FLAGS to record each warp's execution time and print out the result.
```
# Profiling flags
# CC_FLAG += -DWORKLOAD
CC_FLAG += -UWORKLOAD
# NVCC_FLAG += -DWORKLOAD
NVCC_FLAG += -UWORKLOAD
```

### (2) Compile source code
Run the makefile to compile source code and create necessary directories:
```
cd maxflow-cuda
make
```

### (3) Prepare the networks

* 1st DIMACS Challenge synethesized networks (types: Washiongton, Genrmf):
    ```
    % Washiongton
    cd network/DIMACS/washiongton
    cc washington.c -o gengraph

    % Genrmf
    cd network/DIMACS/genrmf
    make
    ```


* SNAP


### (4) Execute the binary
```
Usage: ./maxflow [-h] [-v file type] [-f filename] [-s source] [-t sink]
Options:
	-h		Show this help message and exit
	-v		Specify which kind of file to load
				0: SNAP txt file (default)
				1: SNAP binary file
				2: DIMACS file
	-f filename	Specify the file path (binary or txt)
	-s source	Source node
	-t sink		Sink node
	-a algorithm	Specify the algorithm to use
				0: Thread-centric push-relabel (default)
				1: Vertex-centric push-relabel
```

### (5) Use the scripts

* `maxflow-cuda/bash_run.py`: Execute all files in the given directory and capture the total execution time and each warp's accumulated execution time
    ```
    usage: bash_run.py [-h] [--log LOG] --dir DIR --times TIMES [--stats STATS]

    optional arguments:
    -h, --help     show this help message and exit
    --log LOG      Path to the log file or 'stdout' for console output.
    --dir DIR      Path to the directory where the input files to execute.
    --times TIMES  Path to the file where execution times of each warp will be logged. (Please add the WORKLOAD flag.)
    --stats STATS  Path to the file where statistics of warp execution time will be logged. (Please add the WORKLOAD flag.)
    ```

* `plots/boxplot`: Plot the boxplot of workload
    ```
    usage: boxplot.py [-h] --file FILE --output OUTPUT

    optional arguments:
    -h, --help       show this help message and exit
    --file FILE      Path to the statistics file generated from bash_run.py.
    --output OUTPUT  Path to the output image file.
    ```

## Citation
If this project helps you well, please give us a star or cite our paper.
```
@article{hsieh2024engineering,
  title={Engineering A Workload-balanced Push-Relabel Algorithm for Massive Graphs on GPUs},
  author={Hsieh, Chou-Ying and Lin, Po-Chieh and Kuo, Sy-Yen},
  journal={arXiv preprint arXiv:2404.00270},
  year={2024}
}
``` 
