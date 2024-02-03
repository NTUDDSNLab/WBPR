#include"../include/parallel_graph.cuh"

void global_relabel(int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows, 
                int* cpu_roffsets, int* cpu_rdestinations, int* cpu_flow_idx,
                int *Excess_total, bool *mark, bool *scanned)
{
    for (int u = 0; u < V; u++) {
        // For all (u,v) belonging to E
        for (int i = cpu_offsets[u]; i < cpu_offsets[u + 1]; i++) {
            int v = cpu_destinations[i];

            if (cpu_height[u] > cpu_height[v] + 1) {


                cpu_excess_flow[u] -= cpu_fflows[i];
                cpu_excess_flow[v] += cpu_fflows[i];
                cpu_bflows[i] += cpu_fflows[i];
                cpu_fflows[i] = 0;
            }

        }
    }

    //printf("Prebfs\n");
    // performing backwards bfs from sink and assigning height values with each vertex's BFS tree level
    
    // declaring the Queue 
    std::list<int> Queue;

    // declaring variables to iterate over nodes for the backwards bfs and to store current tree level
    int x,y,current;
    
    // initialisation of the scanned array with false, before performing backwards bfs
    for(int i = 0; i < V; i++)
    {
        scanned[i] = false;
    }

    // Enqueueing the sink and set scan(sink) to true 
    Queue.push_back(sink);
    scanned[sink] = true;
    cpu_height[sink] = 0;

    // bfs routine and assigning of height values with tree level values
    while(!Queue.empty())
    {
        // dequeue
        
        x = Queue.front();
        Queue.pop_front();

        PRINTF("Global relabel: Dequeued: %d\n", x);

        // capture value of current level
        current = cpu_height[x];
        
        // increment current value
        current = current + 1;

        // For all (y,x) belonging to E_f
        // Scan reversed CSR but use the flow in the forward direction 
        for(int i = cpu_roffsets[x]; i < cpu_roffsets[x + 1]; i++)
        {
            y = cpu_rdestinations[i];
            int flow_index = cpu_flow_idx[i];
            PRINTF("Global relabel: (%d, %d)'s flow: %d\n",y, x, cpu_fflows[flow_index]);
            
            if (cpu_fflows[flow_index] > 0) {
                // if y is not scanned
                PRINTF("Global relabel: (%d, %d)'s flow > 0\n",x, y);
                if(scanned[y] == false)
                {
                    // assign current as height of y node
                    cpu_height[y] = current;

                    // mark scanned(y) as true
                    scanned[y] = true;

                    // Enqueue y
                    Queue.push_back(y);
                    PRINTF("Global relabel: Enqueued: %d\n", y);
                }
            }

        }

        // for(y = 0; y < V; y++)
        // {
        //     // for all (y,x) belonging to E_f (residual graph)
        //     if(cpu_rflowmtx[IDX(y,x)] > 0)
        //     {
        //         // if y is not scanned
        //         if(scanned[y] == false)
        //         {
        //             // assign current as height of y node
        //             cpu_height[y] = current;

        //             // mark scanned(y) as true
        //             scanned[y] = true;

        //             // Enqueue y
        //             Queue.push_back(y);
        //         }
        //     }
        // }

    }
    //printf("Pre check\n");
    // declaring and initialising boolean variable for checking if all nodes are relabeled
    bool if_all_are_relabeled = true;

    for(int i = 0; i < V; i++)
    {
        if(scanned[i] == false)
        {
            if_all_are_relabeled = false;
            break;
        }
    }

    // if not all nodes are relabeled
    if(if_all_are_relabeled == false)
    {
        // for all nodes
        for(int i = 0; i < V; i++)
        {
            // if i'th node is not marked or relabeled
            if( !( (scanned[i] == true) || (mark[i] == true) ) )
            {
                // mark i'th node
                mark[i] = true;

                /* decrement excess flow of i'th node from Excess_total
                    * This shows that i'th node is not scanned now and needs to be marked, thereby no more contributing to Excess_total
                    */
                PRINTF("Global relabel: %d is not scanned\n", i);
                *Excess_total = *Excess_total - cpu_excess_flow[i];
                //printf("Global relabel: Excess total: %d\n", *Excess_total);
            }
        }
    }
}


bool checkEnd(int V, int E, int source, int sink, int* cpu_excess_flow) {
    for (int u = 0; u < V; u++) {
        if (u != source && u != sink) {
            if (cpu_excess_flow[u] > 0) {
                return false;
            }
        }
    }
    return true;
}