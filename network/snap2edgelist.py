import argparse

def convert_snap_to_edgelist(input_snap_file):
    unique_vertices = set()
    edges = []

    with open(input_snap_file, 'r') as file:
        snap_edges = file.readlines()

    print("The graph: ", snap_edges[0])


    for edge in snap_edges:
        if edge.startswith("#"):
            continue

        u, v = map(int, edge.split())
        unique_vertices.update([u, v])
        edges.append((u, v))

    sorted_vertices = sorted(unique_vertices)
    is_continuous = len(unique_vertices) == max(unique_vertices) + 1 and min(unique_vertices) == 0

    if not is_continuous:
        vertex_mapping = {original: new for new, original in enumerate(sorted_vertices)}
        adjusted_edges = [(vertex_mapping[u], vertex_mapping[v]) for u, v in edges]
    else:
        adjusted_edges = edges

    return ["{} {} 1".format(u, v) for u, v in adjusted_edges], len(unique_vertices), len(edges), is_continuous

input_snap_file = ''
output_edgelist_file = ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=input_snap_file)
    parser.add_argument('--output', type=str, default=output_edgelist_file)
    args = parser.parse_args()

    input_snap_file = args.input
    output_edgelist_file = args.output

    edgelist, num_nodes, num_edges, is_continuous = convert_snap_to_edgelist(input_snap_file)

    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)
    print("Is continuous:", is_continuous)


    with open(output_edgelist_file, 'w') as file:
        for edge in edgelist:
            file.write("{}\n".format(edge))
