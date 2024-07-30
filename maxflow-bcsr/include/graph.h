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
#include <cmath>
#include <numeric>


// CSR graph representation
class CSRGraph {
public:
  CSRGraph() = default;
  CSRGraph(const std::string &filename);
  ~CSRGraph() = default;
  void buildFromTxtFile(const std::string &filename);
  void buildFromMmioFile(const std::string &filename);
  void buildFromDIMACSFile(const std::string &filename);
  void saveToBinary(const std::string &filename);
  void checkIfContinuous();
  void checkIfLowerTriangle();
  bool loadFromBinary(const std::string &filename);
  bool operator==(const CSRGraph &rhs) const {
    return num_nodes == rhs.num_nodes && num_edges == rhs.num_edges &&
           destinations == rhs.destinations && offsets == rhs.offsets;
  }
  void printGraphStatus() const;

  int num_nodes;
  int num_edges;
  int num_edges_processed;
  int source_node; // For maximum flow - DIMACS
  int sink_node; // For maximum flow - DIMACS
  std::vector<int> destinations;
  std::vector<int> offsets;
  std::vector<int> capacities;

private:
  std::pair<double, double> calDegree() const;
  std::pair<int, int> findMaxMinDegreeNode() const;
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

    int countActiveNodes(void);

    bool push(int u); // Return true if the node u can push flow

    void relabel(int u);

    bool checkPath();

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
    int* backward_idx; // BCSR: index of the reverse edge in the residual graph
    

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
