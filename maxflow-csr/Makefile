CC = g++ -std=c++11
NVCC = nvcc -std=c++11

CC_FLAG = -O3 
NVCC_FLAG = -O3 --cudart shared

# Profiling flags
CC_FLAG += -DWORKLOAD
# CC_FLAG += -UWORKLOAD
NVCC_FLAG += -DWORKLOAD
# NVCC_FLAG += -UWORKLOAD

# run make all to create ./maxflow executable
all : maxflow

# run make debug to create ./maxflowdbg executable, which is the debug build to use in cuda-gdb and cuda-memcheck
debug : maxflowdbg

maxflow : obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o obj/graph.o obj/mmio.o obj/global_relabel_gpu.o
	$(NVCC) $(NVCC_FLAG) obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o obj/graph.o obj/mmio.o obj/global_relabel_gpu.o -o maxflow

maxflowdbg : obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o obj/global_relabel_gpu.o
	$(NVCC) $(NVCC_FLAG) obj/graph_s.o obj/io_par.o obj/main.o obj/preflow.o obj/push_relabel_kernel.o obj/push_relabel.o obj/global_relabel.o obj/global_relabel_gpu.o -g -G -o maxflowdbg

obj/main.o : src/main.cu obj
	$(NVCC) $(NVCC_FLAG) -c src/main.cu -o obj/main.o

obj/graph_s.o : src/graph_s.cpp obj
	$(CC) $(CC_FLAG) -c src/graph_s.cpp -o obj/graph_s.o

obj/io_par.o : src/io_par.cu obj
	$(NVCC) $(NVCC_FLAG) -c src/io_par.cu -o obj/io_par.o

obj/graph.o : src/graph.cpp obj
	$(CC) $(CC_FLAG) -c src/graph.cpp -o obj/graph.o

obj/mmio.o : src/mmio.c obj
	$(CC) $(CC_FLAG) -c src/mmio.c -o obj/mmio.o

obj/preflow.o : src/preflow.cu obj
	$(NVCC) $(NVCC_FLAG) -c src/preflow.cu -o obj/preflow.o 

obj/push_relabel.o : src/push_relabel.cu obj
	$(NVCC) $(NVCC_FLAG) -c src/push_relabel.cu -o obj/push_relabel.o 

obj/push_relabel_kernel.o : src/push_relabel_kernel.cu obj
	$(NVCC) $(NVCC_FLAG) -c src/push_relabel_kernel.cu -o obj/push_relabel_kernel.o 

obj/global_relabel.o : src/global_relabel.cu obj
	$(NVCC) $(NVCC_FLAG) -c src/global_relabel.cu -o obj/global_relabel.o

obj/global_relabel_gpu.o : src/global_relabel_gpu.cu obj
	$(NVCC) $(NVCC_FLAG) -c src/global_relabel_gpu.cu -o obj/global_relabel_gpu.o

obj :
	mkdir obj

# run make clean to clean obj files and the executable(s)
clean :

	if [ obj ] ; \
	then \
		rm obj/*.o ; \
		rmdir obj ; \
	fi ; \
	if [ maxflow ] ; \
	then \
		rm ./maxflow ; \
	fi ;
	if [ maxflowdbg ] ; \
	then \
		rm ./maxflowdbg ; \
	fi ;
