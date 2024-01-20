import argparse

def convert_to_edgelist(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    start_from_zero = False
    # Check if the vertex ID starts from 1 or 0
    for line in lines:
        if line.startswith('n'):
            _, v, types = line.split()
            print("Vertex ID starts from", v)
            print("Type of graph:", types)
            if (types == 's' and v == '1'):
                start_from_zero = False
                print("Vertex ID starts from 1")
            else:
                start_from_zero = True
                print("Vertex ID starts from 0")
            break

    with open(output_file, 'w') as file:
        for line in lines:
            if line.startswith('a'):
                _, u, v, w = line.split()
                if (not start_from_zero):
                    u = str(int(u) - 1)
                    v = str(int(v) - 1)
                file.write(f"{u} {v} {w}\n")

input_genrmf_file = 'path/to/your/GENRMF_file.txt'  # Replace with your GENRMF file path
output_edgelist_file = 'path/to/your/edgelist_file.txt'  # Replace with your desired output path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=input_genrmf_file)
    parser.add_argument('--output', type=str, default=output_edgelist_file)
    args = parser.parse_args()

    input_genrmf_file = args.input
    output_edgelist_file = args.output

    convert_to_edgelist(input_genrmf_file, output_edgelist_file)
