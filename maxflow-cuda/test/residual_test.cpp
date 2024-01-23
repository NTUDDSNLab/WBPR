#include "../include/graph.h"
#include <iostream>
#include <string.h>

using namespace std;

int main(int argc, char* argv[])
{
    // string mmio_file = string(argv[1]);
    // string csr_binary_file = mmio_file + ".bin";
    // /* Test ResidualGraph */
    // CSRGraph csr;
    // if (csr.loadFromBinary(csr_binary_file)) {
    //     cout << "Loading " << mmio_file << " from binary file... " << endl;
    // } else {
    //     cout << "No binary file found! ..." << endl;
    //     exit(1);
    // }
    
    CSRGraph csr;
    csr.buildFromTxtFile(string(argv[1]));
    /* Print csr */
    cout << "CSRGraph:" << endl;
    cout << "num_nodes: " << csr.num_nodes << endl;
    cout << "num_edges: " << csr.num_edges << endl;
    cout << "offsets: ";
    for (int i = 0; i < csr.offsets.size(); ++i)
        cout << csr.offsets[i] << " ";
    cout << endl;
    cout << "destinations: ";
    for (int i = 0; i < csr.destinations.size(); ++i)
        cout << csr.destinations[i] << " ";
    cout << endl;
    cout << "capacities: ";
    for (int i = 0; i < csr.capacities.size(); ++i)
        cout << csr.capacities[i] << " ";
    cout << endl;

    
    ResidualGraph res(csr);

   
    /* Print res */
    cout << "ResidualGraph:" << endl;
    cout << "forward_offsets: ";
    for (int i = 0; i < res.forward_offsets.size(); ++i)
        cout << res.forward_offsets[i] << " ";
    cout << endl;
    cout << "forward_destinations: ";
    for (int i = 0; i < res.forward_destinations.size(); ++i)
        cout << res.forward_destinations[i] << " ";
    cout << endl;
    cout << "forward_capacities: ";
    for (int i = 0; i < res.forward_capacities.size(); ++i)
        cout << res.forward_capacities[i] << " ";
    cout << endl;
    cout << "backward_offsets: ";
    for (int i = 0; i < res.backward_offsets.size(); ++i)
        cout << res.backward_offsets[i] << " ";
    cout << endl;
    cout << "backward_destinations: ";
    for (int i = 0; i < res.backward_destinations.size(); ++i)
        cout << res.backward_destinations[i] << " ";
    cout << endl;
    cout << "backward_capacities: ";
    for (int i = 0; i < res.backward_capacities.size(); ++i)
        cout << res.backward_capacities[i] << " ";
    cout << endl;



    return 0;
}
