from mpi4py import MPI
import numpy as np
import time

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Mapping user input to matrix sizes
matrix_sizes = {
    1: 256, 2: 512, 3: 1024, 4: 2048, 5: 4096,
    6: 4608, 7: 5120, 8: 6144, 9: 6656, 10: 7168,
    11: 7680, 12: 8192, 13: 8704, 14: 9216, 15: 9728,
    16: 10240, 17: 10752, 18: 11264, 19: 11776, 20: 12288,
    21: 12800, 22: 13312, 23: 13824, 24: 14336, 25: 14848
}

# Root process asks for matrix size
if rank == 0:
    print("\nMatrix Multiplication Benchmark")
    print("Select a matrix size (1-25):")
    for i in range(1, 26):
        print(f"{i}: {matrix_sizes[i]} x {matrix_sizes[i]}")
    selection = int(input("\nEnter your choice (1-25): ").strip())
    matrix_size = matrix_sizes.get(selection, 512)
else:
    matrix_size = None

# Broadcast the selected matrix size to all processes
matrix_size = comm.bcast(matrix_size, root=0)

# Divide rows among processes
rows_per_process = matrix_size // size
remaining_rows = matrix_size % size

# Create matrices A and B on the root process
A = None
B = None
if rank == 0:
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)

# Allocate local storage for each process
local_rows = rows_per_process + (1 if rank < remaining_rows else 0)
A_local = np.zeros((local_rows, matrix_size))

# Scatter rows of A
counts = [rows_per_process + (1 if i < remaining_rows else 0) for i in range(size)]
displacements = [sum(counts[:i]) for i in range(size)]
comm.Scatterv([A, [count * matrix_size for count in counts], displacements, MPI.DOUBLE], A_local, root=0)

# Broadcast B to all processes
B = comm.bcast(B, root=0)

# Perform local matrix multiplication
start_time = time.time()
C_local = np.dot(A_local, B)
computation_time = time.time() - start_time

# Gather computation times and results on root
computation_times = comm.gather(computation_time, root=0)
C = None
if rank == 0:
    C = np.zeros((matrix_size, matrix_size))
comm.Gatherv(C_local, [C, [count * matrix_size for count in counts], displacements, MPI.DOUBLE], root=0)

# Calculate benchmarking data
if rank == 0:
    total_operations = 2 * (matrix_size ** 3)  # 2 * n^3 FLOPS
    max_time = max(computation_times)  # Max time across processes
    flops = total_operations / max_time
    gflops = flops / 1e9
    tops = flops / 1e12
    avg_gflops_per_rank = gflops / size

    # Print results
    print("\n==================== Matrix Multiplication Benchmark ====================")
    print(f"Matrix size: {matrix_size} x {matrix_size}")
    print(f"Number of processes: {size}")
    print(f"Total computation time (max across nodes): {max_time:.4f} seconds")
    print(f"Total operations: {total_operations:,} FLOPs")
    print(f"Performance: {flops:.2f} FLOPs")
    print(f"Performance: {gflops:.2f} GFLOPs")
    print(f"Performance: {tops:.4f} TFLOPs")
    print(f"Average GFLOPs per rank: {avg_gflops_per_rank:.2f}")
    print("=========================================================================")

    # Display per-rank data
    print("\nPer-Process Computation Times:")
    for i, t in enumerate(computation_times):
        print(f"Process {i}: {t:.4f} seconds")

