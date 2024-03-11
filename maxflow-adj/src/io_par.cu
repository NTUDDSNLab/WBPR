#include"../include/parallel_graph.cuh"

using namespace std;

void
readFromDIMACSFormat(std::string filename, ull *V, ull *E, ull *source, ull *sink, int **cpu_height, int **cpu_excess_flow, int **cpu_adjmtx, int **cpu_rflowmtx)
{
    cout << "Reading from DIMACS format file: " << filename << endl;
    // declaring file pointer to read edgelist
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;

    while (getline(file, line)) {
        if (line[0] == 'c') {
            // Comment line, ignore
            continue;
        } else if (line[0] == 'p') {
            // Problem line
            std::istringstream iss(line);
            char p;
            std::string format;
            iss >> p >> format >> *V >> *E;

            // allocating host memory
            *cpu_adjmtx = (int *)malloc(*V**V*sizeof(int));
            *cpu_rflowmtx = (int *)malloc(*V**V*sizeof(int));

            for (ull i = 0; i < *V; i++) {
                for (ull j = 0; j < *V; j++) {
                    (*cpu_adjmtx)[IDX(i,j)] = 0;
                    (*cpu_rflowmtx)[IDX(i,j)] = 0;
                }
            }


        } 
            else if (line[0] == 'n') {
            // Node designation line (source or sink)
            std::istringstream iss(line);
            char n;
            int node_id;
            char node_type;
            iss >> n >> node_id >> node_type;
            if (node_type == 's') {
                *source = node_id - 1;  // Convert to 0-based index
            } else if (node_type == 't') {
                *sink = node_id - 1;    // Convert to 0-based index
            } 
        }      
            else if (line[0] == 'a') {
            // Edge descriptor line
            std::istringstream iss(line);
            char a;
            int u, v, capacity;
            iss >> a >> u >> v >> capacity;

            // Note: DIMACS format uses 1-based indexing, while C++ uses 0-based indexing
            (*cpu_adjmtx)[IDX((u-1),(v-1))] = capacity;
            (*cpu_rflowmtx)[IDX((u-1),(v-1))] = capacity;
        }
    }
}


void print(ull *V,int *cpu_height, int *cpu_excess_flow, int *cpu_rflowmtx, int *cpu_adjmtx)
{
    printf("\nHeight :");
    for(ull i = 0; i < *V; i++)
    {
        printf("%d ",cpu_height[i]);
    }

    printf("\nExcess flow :");
    for(ull i = 0; i < *V; i++)
    {
        printf("%d ",cpu_excess_flow[i]);
    }

    printf("\nRflow mtx :\n");
    for(ull i = 0; i < *V; i++)
    {
        for(ull j = 0; j < *V; j++)
        {
            printf("%d ", cpu_rflowmtx[IDX(i,j)]);
        }
        printf("\n");
    }

    printf("\nAdj mtx :\n");
    for(ull i = 0; i < *V; i++)
    {
        for(ull j = 0; j < *V; j++)
        {
            printf("%d ", cpu_adjmtx[IDX(i,j)]);
        }
        printf("\n");
    }
}