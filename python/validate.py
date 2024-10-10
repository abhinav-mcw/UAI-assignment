import numpy as np
import argparse

def read_binary_file(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data

def compute_difference(file1, file2):
    data1 = read_binary_file(file1)
    data2 = read_binary_file(file2)

    if data1.shape != data2.shape:
        raise ValueError(f"Files have different shapes: {data1.shape} vs {data2.shape}")
    diff = np.abs(data1 - data2)
    min = np.min(diff)
    max = np.max(diff)
    mean = np.mean(diff)
    print("Min difference: ", min)
    print("Max difference: ", max)
    print("Mean difference: ", mean)
    if mean < 0.001:
        print("Files are identical")

def main():
    parser = argparse.ArgumentParser(description="Compare two binary files and find differences.")
    parser.add_argument("file1", help="Path to the first binary file")
    parser.add_argument("file2", help="Path to the second binary file")

    args = parser.parse_args()

    file1 = args.file1
    file2 = args.file2

    try:
        compute_difference(file1, file2)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()