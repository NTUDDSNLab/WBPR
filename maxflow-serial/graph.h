/*
 * Author: Andrew
 * Date: January 1, 2024
 * Description:
 *   This file contains the definition of CSRGraph
 *
 * Appendices:
 * 1. shout out to slimon writing most of the code.
 * 2. num_edges_processed is to verify #edges.
 */
#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// CSR graph representation
class CSRGraph {
public:
  CSRGraph() = default;
  CSRGraph(const std::string &filename);
  ~CSRGraph() = default;
  void buildFromTxtFile(const std::string &filename);
  void buildFromMmioFile(const std::string &filename);
  void saveToBinary(const std::string &filename);
  void checkIfContinuous();
  void checkIfLowerTriangle();
  bool loadFromBinary(const std::string &filename);
  bool operator==(const CSRGraph &rhs) const {
    return num_nodes == rhs.num_nodes && num_edges == rhs.num_edges &&
           destinations == rhs.destinations && offsets == rhs.offsets;
  }

  int num_nodes;
  int num_edges;
  int num_edges_processed;
  std::vector<int> destinations;
  std::vector<int> offsets;
  std::vector<int> capacities;
};

// Residual graph representation
class ResidualGraph {
public:
    ResidualGraph() {}
    ResidualGraph(const CSRGraph &graph) {
        buildFromCSRGraph(graph);
    }

    void buildFromCSRGraph(const CSRGraph &graph);

    void print() const;

    void preflow(int source);

    /* Find the active node with the MAX height, return nodeID, or -1 if not find */
    int findActiveNode(void);

    bool push(int u); // Return true if the node u can push flow

    void relabel(int u);

    void maxflow(int source, int sink); 

    // Other methods and members...

    int num_nodes;
    int num_edges;

    int source;
    int sink;

    int* offsets;
    int* destinations;
    int* capacities;
    int* forward_flows; // cf(u, v)
    int* backward_flows; // cf(v, u)

    int* roffsets;
    int* rdestinations;
    int* flow_index;
    

    int* heights;
    int* excesses;

    int Excess_total;
    // std::vector<int> offsets;
    // std::vector<int> destinations;
    // std::vector<int> capacities; 
    // std::vector<int> flows; // cf(u, v)

    // std::vector<int> roffsets;
    // std::vector<int> rdestinations;
    // std::vector<int> rflows; // cf(v, u)
};
