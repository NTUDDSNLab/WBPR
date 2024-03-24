def find_largest_number():
    left_max = 0
    middle_max = 0
    with open('edgelist.txt', 'r') as f:
        edges = f.readlines()
    for edge in edges:
        u, v, _ = map(int, edge.split())
        left_max = max(left_max, u)
        middle_max = max(middle_max, v)
    return left_max, middle_max

if __name__ == '__main__':
    left, middle = find_largest_number()
    print("left max: ", left)
    print("middle max: ", middle)