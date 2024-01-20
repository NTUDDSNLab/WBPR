import argparse

def convert_to_edgelist(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            if line.startswith('a'):
                _, u, v, w = line.split()
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
