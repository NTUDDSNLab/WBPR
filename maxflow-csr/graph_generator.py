import networkx as nx
import random
import argparse

def randomGenerator(num_nodes, num_edges, max_weight):
    # Create a directed graph
    G = nx.DiGraph()

    G.add_nodes_from(range(num_nodes))

    for _ in range(num_edges):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        while dst == src:  # Ensure the destination is different from the source
            dst = random.randint(0, num_nodes - 1)
        weight = random.randrange(max_weight)  # Assign a random weight between 0.1 and 10
        G.add_edge(src, dst, weight=weight)

    # Select source and sink vertices for maximum flow algorithm
    source = 0  # First node as the source
    sink = num_nodes - 1  # Last node as the sink

    # Generate the edge list
    edge_list = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]

    # Limiting the output to first 20 edges for display, as the full list is too large to show
    return edge_list, source, sink



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate a random graph')
    parser.add_argument('--nodes', type=int, default=1000,
                        help='Number of nodes in the graph')
    parser.add_argument('--edges', type=int, default=200000,
                        help='Number of edges in the graph')
    parser.add_argument('--output', type=str, default='edgelist.txt',
                        help='Output file to store the generated graph')
    parser.add_argument('--weight-range', type=int, default=10,
                        help='Range of weights for the edges')


    args = parser.parse_args()

    edgelist, source, sink = randomGenerator(args.nodes, args.edges, args.weight_range)

    # Store the graph in a file and print the source and sink vertices
    with open('edgelist.txt', 'w') as f:
        for edge in edgelist:
            f.write('{} {} {}\n'.format(edge[0], edge[1], edge[2]))
    
    print('Source: {}'.format(source))
    print('Sink: {}'.format(sink))

    