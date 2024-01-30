/*
 * Author: Andrew
 * Date: January 1, 2024
 * Description:
 *   This file contains the definition of CSRGraph.
 * About to add a function so it could read mmio.
 *
 * Appendices:
 * 1. shout out to slimon writing most of the code.
 */
#include "graph.h"
#include "assert.h"
#include "mmio.h"

CSRGraph::CSRGraph(const std::string &filename) {
  if (filename.substr(filename.find_last_of(".") + 1) == "txt") {
    buildFromTxtFile(filename);
  } else if (filename.substr(filename.find_last_of(".") + 1) == "mmio") {
    std::cout << "buildFromMmioFile: " << filename << std::endl;
    buildFromMmioFile(filename);
  } else {
    std::cerr << "Unsupported file type for: " << filename << '\n';
    exit(1);
  }
}

void CSRGraph::buildFromTxtFile(const std::string &filename) {
  std::ifstream file(filename);
  if (file.fail()) {
    fprintf(stderr, "\"%s\" does not exist!\n", filename.c_str());
    exit(1);
  }
  std::string line;
  std::unordered_map<int, std::vector<int>> adjacency_list;
  int cnt = 0;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    if (line.find("# Nodes:") != std::string::npos)
      sscanf(line.c_str(), "# Nodes: %d Edges: %d", &num_nodes, &num_edges);

    if (ss.str()[0] == '#')
      continue;
    int from, to;
    ss >> from >> to;
    adjacency_list[from].push_back(to);
    cnt++;
  }

  // num_nodes = adjacency_list.size();
  offsets.push_back(0);
  for (int i = 0; i < num_nodes; ++i) {
    // some nodes have no out edges
    if (adjacency_list.count(i)==0) {
      offsets.push_back(offsets.back());
      continue;
    }
    sort(adjacency_list[i].begin(), adjacency_list[i].end());
    for (int neighbor : adjacency_list[i]) {
      destinations.push_back(neighbor);
      capacities.push_back(1);
    }
    offsets.push_back(destinations.size());
  }

  num_edges_processed = cnt;
}

void CSRGraph::buildFromMmioFile(const std::string &filename) {
  FILE *f;
  int ret_code;
  MM_typecode matcode;
  int M, N, nz; // rows, columns, entries. #vertex, #vertex, #edges in graph
  int *I, *J;
  double *val;

  if ((f = fopen(filename.c_str(), "r")) == NULL) {
    fprintf(stderr, "\"%s\" does not exist!\n", filename.c_str());
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */
  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode)) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
    printf("read mtx size error.");
    exit(1);
  }

  num_nodes = M;
  num_edges = nz;

  I = (int *)malloc(nz * sizeof(int));
  J = (int *)malloc(nz * sizeof(int));
  val = (double *)malloc(nz * sizeof(double));

  std::unordered_map<int, std::vector<int>> adjacency_list;
  int cnt = 0;

  for (int i = 0; i < nz; i++) {
    if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]) != 3) {
      fprintf(stderr, "Error reading values from file.\n");
      exit(EXIT_FAILURE);
    }
    I[i]--; /* adjust from 1-based to 0-based */
    J[i]--;
    assert(I[i] >= 0 && J[i] >= 0);
    assert(I[i] > J[i]);
    adjacency_list[I[i]].push_back(J[i]);
    cnt++;
  }

  if (f != stdin)
    fclose(f);

  offsets.push_back(0);
  for (int i = 0; i < num_nodes; ++i) {
    // some nodes have no out edges
    if (!adjacency_list.count(i)) {
      // std::cout << "node " << i << " has no friend." << std::endl;
      offsets.push_back(offsets.back());
      continue;
    }
    sort(adjacency_list[i].begin(), adjacency_list[i].end());
    for (int neighbor : adjacency_list[i])
      destinations.push_back(neighbor);
    offsets.push_back(destinations.size());
  }

  num_edges_processed = cnt;
  std::cout << "nodes: " << num_nodes << std::endl;
  std::cout << "edges: " << num_edges << std::endl;
  std::cout << "edges processed: " << num_edges_processed << std::endl;

  free(I);
  free(J);
  free(val);
}

void CSRGraph::saveToBinary(const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Could not open file for writing: " << filename << '\n';
    return;
  }
  file.write(reinterpret_cast<const char *>(&num_nodes), sizeof(num_nodes));
  file.write(reinterpret_cast<const char *>(&num_edges), sizeof(num_edges));
  file.write(reinterpret_cast<const char *>(&num_edges_processed),
             sizeof(num_edges_processed));
  file.write(reinterpret_cast<const char *>(destinations.data()),
             destinations.size() * sizeof(int));
  file.write(reinterpret_cast<const char *>(offsets.data()),
             offsets.size() * sizeof(int));
}

void CSRGraph::checkIfContinuous() {
  // result: every node in amazon0302 has at least one edge.

  std::cout << "destination nodes: " << destinations.size() << std::endl;

  std::vector<int> nodeNoOut;
  int prev_off = -1, i = 0;
  for (int &off : offsets) {
    if (prev_off == off) {
      // std::cout << "node "<< i << " has no out."<< std::endl;
      nodeNoOut.push_back(i); // already 1-index
    }
    prev_off = off;
    i++;
  }

  std::set<int> nodeNoIn;
  for (int i = 1; i <= num_nodes; ++i) {
    nodeNoIn.insert(i);
  }
  for (int num : destinations) {
    auto it = nodeNoIn.find(num + 1); // back to 1-index
    if (it != nodeNoIn.end()) {
      nodeNoIn.erase(it);
    }
  }

  std::set<int> intersectionSet;
  std::set_intersection(
      nodeNoIn.begin(), nodeNoIn.end(), nodeNoOut.begin(), nodeNoOut.end(),
      std::inserter(intersectionSet, intersectionSet.begin()));

  std::cout << "Intersection: ";
  for (int num : intersectionSet) {
    std::cout << num << " ";
  }
  std::cout << std::endl;
}

void CSRGraph::checkIfLowerTriangle() {
  // check if lower triangle
  for (int u = 0; u < this->num_nodes; u++) {
    if (this->offsets[u] == this->offsets[u + 1])
      continue;
    // std::cout << u << ": ";
    int prev = -1;
    for (int j = this->offsets[u]; j < this->offsets[u + 1]; j++) {
      int v = this->destinations[j];
      // std::cout << v << " ";
      assert(v > prev);
      assert(v < u);
      prev = v;
    }
    // std::cout << std::endl;
  }
}

bool CSRGraph::loadFromBinary(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Could not open file for reading: " << filename << '\n';
    return false;
  }
  file.read(reinterpret_cast<char *>(&num_nodes), sizeof(num_nodes));
  file.read(reinterpret_cast<char *>(&num_edges), sizeof(num_edges));
  file.read(reinterpret_cast<char *>(&num_edges_processed),
            sizeof(num_edges_processed));
  destinations.resize(num_edges_processed);
  offsets.resize(num_nodes + 1);
  file.read(reinterpret_cast<char *>(destinations.data()),
            destinations.size() * sizeof(int));
  file.read(reinterpret_cast<char *>(offsets.data()),
            offsets.size() * sizeof(int));
  // Initialize capacities
  capacities.resize(num_edges_processed, 1);
  std::cout << "nodes: " << num_nodes << std::endl;
  std::cout << "edges: " << num_edges << std::endl;
  std::cout << "num_edges_processed: " << num_edges_processed << std::endl;
  return true;
}


void ResidualGraph::buildFromCSRGraph(const CSRGraph &graph) {
  num_nodes = graph.num_nodes;
  num_edges = graph.num_edges;

  /* Allocate offsets, destinations, capacities, flows, roffsets, rdestinations, rflows */
  offsets = (int *)malloc(sizeof(int) * (num_nodes + 1));
  destinations = (int *)malloc(sizeof(int) * num_edges);
  capacities = (int *)malloc(sizeof(int) * num_edges);
  forward_flows = (int *)malloc(sizeof(int) * num_edges);
  roffsets = (int *)malloc(sizeof(int) * (num_nodes + 1));
  rdestinations = (int *)malloc(sizeof(int) * num_edges);
  backward_flows = (int *)malloc(sizeof(int) * num_edges);
  flow_index = (int *)malloc(sizeof(int) * num_edges);

  heights = (int *)malloc(sizeof(int) * num_nodes);
  excesses = (int *)malloc(sizeof(int) * num_nodes);

  for (int i = 0; i < num_nodes; i++) {
    heights[i] = 0;
    excesses[i] = 0;
  }

  // Initialize offset vectors
  for (int i = 0; i < num_nodes + 1; ++i) {
    offsets[i] = graph.offsets[i];
    roffsets[i] = 0;
  }
  for (int i = 0; i < num_edges; ++i) {
    destinations[i] = graph.destinations[i];
    capacities[i] = graph.capacities[i];
    forward_flows[i] = graph.capacities[i]; // The initial residual flow is the same as capacity, not 0
    backward_flows[i] = 0;
  }

  

  std::vector<int> backward_counts(num_nodes, 0);

  // Count the number of edges for each node to prepare the offset vectors
  for (int i = 0; i < num_nodes; ++i) {
      for (int j = graph.offsets[i]; j < graph.offsets[i + 1]; ++j) {
          backward_counts[graph.destinations[j]]++;
      }
  }
  
  // Convert counts to actual offsets
  for (int i = 1; i <= num_nodes; ++i) {
      roffsets[i] = roffsets[i - 1] + backward_counts[i - 1];
  }

  // Initialize backward count vector
  backward_counts.clear();
  for (int i = 0; i < num_nodes; ++i) {
      backward_counts[i] = 0;
  }


  // Fill forward and backward edges
  for (int i = 0; i < num_nodes; ++i) {
      for (int j = graph.offsets[i]; j < graph.offsets[i + 1]; ++j) {
          int dest = graph.destinations[j];

          // Corresponding backward edge
          int backward_index = roffsets[dest] + backward_counts[dest];
          rdestinations[backward_index] = i;
          backward_counts[dest]++;
      }
  }

  // Initialize flow index
  for (int u = 0; u < num_nodes; u++) {
    for (int i = roffsets[u]; i < roffsets[u + 1]; i++) {
      int v = rdestinations[i];

      // Find the forward edge index
      for (int j = offsets[v]; j < offsets[v + 1]; j++) {
        if (destinations[j] == u) {
          flow_index[i] = j;
          break;
        }
      }
    }
  }


}

void ResidualGraph::print() const
{
  printf("Residual graph:\n");
  printf("Offsets: ");
  for (int i=0; i < num_nodes + 1; i++) {
      printf("%d ", offsets[i]);
  }
  printf("\n");
  printf("Destinations: ");
  for (int i=0; i < num_edges; i++) {
      printf("%d ", destinations[i]);
  }
  printf("\n");
  printf("Capacities: ");
  for (int i=0; i < num_edges; i++) {
      printf("%d ", capacities[i]);
  }
  printf("\n");
  printf("Forward Flow: ");
  for (int i=0; i < num_edges; i++) {
      printf("%d ", forward_flows[i]);
  }

  printf("\n");
  printf("Backward Flows: ");
  for (int i=0; i < num_edges; i++) {
      printf("%d ", backward_flows[i]);
  }

  printf("\n");
  printf("Roffsets: ");
  for (int i=0; i < num_nodes + 1; i++) {
      printf("%d ", roffsets[i]);
  }
  printf("\n");
  printf("Rdestinations: ");
  for (int i=0; i < num_edges; i++) {
      printf("%d ", rdestinations[i]);
  }
  printf("\n");
  printf("Flow Index: ");
  for (int i=0; i < num_edges; i++) {
      printf("%d ", flow_index[i]);
  }
  printf("\n");

  printf("Heights: ");
  for (int i=0; i < num_nodes; i++) {
      printf("%d ", heights[i]);
  }
  printf("\n");
  printf("Excesses: ");
  for (int i=0; i < num_nodes; i++) {
      printf("%d ", excesses[i]);
  }
  printf("\n");
}


void
ResidualGraph::preflow(int source)
{
  heights[source] = num_nodes; // Make the height of source equal to number of vertices
  Excess_total = 0;

  // Initialize preflow
  for (int i = offsets[source]; i < offsets[source + 1]; ++i) {
    int dest = destinations[i];
    int cap = capacities[i];

    excesses[dest] = cap;
    forward_flows[i] = 0; // residualFlow[(source, dest)] = 0
    backward_flows[i] = cap; // residualFlow[(dest, source)] = cap
  }
}

bool
ResidualGraph::push(int v)
{
  // Find the outgoing edge (v, w) in foward edge with h(v) = h(w) + 1
  for (int i = offsets[v]; i < offsets[v + 1]; ++i) {
    int w = destinations[i];
    if (heights[v] == heights[w] + 1) {
      // Push flow
      int flow = std::min(excesses[v], forward_flows[i]);
      forward_flows[i] -= flow;
      backward_flows[i] += flow;
      excesses[v] -= flow;
      excesses[w] += flow;
      return true;
    }
  }

  // Find the outgoing edge (v, w) in backward edge with h(v) = h(w) + 1
  for (int i = roffsets[v]; i < roffsets[v+1]; ++i) {
    int w = rdestinations[i];
    if (heights[v] == heights[w] + 1) {
      // Push flow
      int push_index = flow_index[i];
      int flow = std::min(excesses[v], backward_flows[push_index]);
      backward_flows[push_index] -= flow;
      forward_flows[push_index] += flow;
      excesses[v] -= flow;
      excesses[w] += flow;
      return true;
    }
  }

  return false;
}

void
ResidualGraph::relabel(int u)
{
  heights[u]+=1;
}

int
ResidualGraph::findActiveNode(void)
{
  int max_height = num_nodes;
  int return_node = -1;
  for (int i = 0; i < num_nodes; ++i) {
    if (excesses[i] > 0 && i != source && i != sink) {
      if (heights[i] < max_height) {
        max_height = heights[i];
        return_node = i;
      }
    }
  }
  return return_node;
}


void
ResidualGraph::maxflow(int source, int sink)
{
  this->source = source;
  this->sink = sink;

  preflow(source);

  printf("After preflow\n");

  int active_node = findActiveNode();

  while(active_node != -1) {
    /* If there is an outgoing edge (v, w) of v in Gf with h(v) = h(w) + 1 */
    if (!push(active_node)) {
      relabel(active_node);
    }
    active_node = findActiveNode();
  }


  /* Calculate Max flow */
  /* Sum all all rflow(u, sink)*/
  printf("Max flow: %d\n", excesses[sink]);

}