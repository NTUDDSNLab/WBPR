#include"../include/parallel_graph.cuh"

void preflow(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, std::vector<int> *offsets, std::vector<int> *destinations, std::vector<int>* capacities, std::vector<int>* flows, std::vector<int>* roffsets, std::vector<int>* rdestinations, std::vector<int>* rflows, int *Excess_total)
{
    // initialising height values and excess flow, Excess_total values
    for(int i = 0; i < V; i++)
    {
        cpu_height[i] = 0; 
        cpu_excess_flow[i] = 0;
    }
    
    cpu_height[source] = V;
    *Excess_total = 0;

    // pushing flow in all edges going out from the source node
    for(int i = offsets->at(source);  i < offsets->at(source + 1); i++) {
        // pushing out of source node
        // cpu_rflowmtx[IDX(source,i)] = 0;
        
        flows->at(i) = 0; // cpu_rflowmtx[IDX(source,i)] = 0;
        
        /* Updating the residual flow value on the back edge 
            u_f(x,s) = u_xs + u_sx
        */
        // cpu_rflowmtx[IDX(i,source)] = cpu_adjmtx[IDX(source,i)] + cpu_adjmtx[IDX(i,source)];
        int neighborID = destinations->at(i);
        std::cout << "neighborID: " << neighborID << std::endl;

        int back_edge_index = -1;
        for (int j = offsets->at(neighborID); j < offsets->at(neighborID + 1); j++) {
            if (destinations->at(j) == source) {
                back_edge_index = j;
                break;
            }
        }

        // Find rflow[i, source]
        int idx = -1;
        for (int j = roffsets->at(neighborID); j < roffsets->at(neighborID + 1); j++) {
            if (rdestinations->at(j) == source) {
                idx = j;
                break;
            }
        }

        std::cout << "back_edge_index: " << back_edge_index << std::endl;

        if (back_edge_index == -1) {
            rflows->at(idx) = capacities->at(i);
        } else {
            rflows->at(idx) = capacities->at(i) + capacities->at(back_edge_index);
        }

        // updating the excess flow value of the node flow is pushed to, from the source
        // cpu_excess_flow[i] = cpu_adjmtx[IDX(source,i)];
        cpu_excess_flow[neighborID] = capacities->at(i);

        // update Excess_total value with the new excess flow value of the node flow is pushed to
        // *Excess_total += cpu_excess_flow[i];
        *Excess_total += cpu_excess_flow[neighborID];
    }


    // for(int i = 0; i < V; i++)
    // {
    //     // for all (source,i) belonging to E :
    //     if(cpu_adjmtx[IDX(source,i)] > 0)
    //     {
    //         // pushing out of source node
    //         cpu_rflowmtx[IDX(source,i)] = 0;
            
    //         /* updating the residual flow value on the back edge
    //          * u_f(x,s) = u_xs + u_sx
    //          * The capacity of the back edge is also added to avoid any push operation back to the source 
    //          * This avoids creating a race condition, where flow keeps travelling to and from the source
    //          */
    //         cpu_rflowmtx[IDX(i,source)] = cpu_adjmtx[IDX(source,i)] + cpu_adjmtx[IDX(i,source)];
            
    //         // updating the excess flow value of the node flow is pushed to, from the source
    //         cpu_excess_flow[i] = cpu_adjmtx[IDX(source,i)];

    //         // update Excess_total value with the new excess flow value of the node flow is pushed to
    //         *Excess_total += cpu_excess_flow[i];
    //     } 
    // }

}