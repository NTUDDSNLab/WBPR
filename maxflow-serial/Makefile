CXX=g++

PROG=maxflow-serial
SRCS=main.cpp graph.cpp mmio.cpp
OBJS=$(SRCS:.cpp=.o)

CXXFLAGS=-O0 -g -Wall -Wextra -std=c++11 -fopenmp -march=native -mtune=native -DNDEBUG

all: $(PROG)

$(PROG): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(PROG) $(OBJS)

.PHONY: all clean
