from mpi4py import MPI
import numpy as np
import time
import sys
import psutil

# ASCII Art Line Width
LINE_WIDTH = 70

# Function to check memory and detect if swap is used
def check_memory():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    if swap.used > 0:
        return False, f"[ERROR] Program terminated: Swap memory is in use (Swap used: {swap.used // (1024 ** 2)} MB). Ensure sufficient RAM is available."
    return True, f"[INFO] Memory Check Passed: RAM available {memory.available // (1024 ** 2)} MB."

# Function to create ASCII boxed sections
def create_ascii_box(title, content, width=LINE_WIDTH):
    title_line = f"+{'-' * (width - 2)}+"
    box = [title_line, f"| {title.center(width - 4)} |", title_line]
    for line in content:
        box.append(f"| {line.ljust(width - 4)} |")
    box.append(title_line)
    return "\n".join(box)

# Function to format computation times
def format_times(times):
    formatted = []
    for i, time in enumerate(times):
        formatted.append(f"[Rank {i}] Time: {time:.4f} seconds")
    return formatted

# Parse matrix size from command-line arguments
matrix_size_option = int(sys.argv[1])
matrix_sizes = {
    1: 256, 2: 512, 3: 1024, 4: 2048, 5: 4096,
    6: 4608, 7: 5120, 8: 6144, 9: 6656, 10: 7168,
    11: 8192, 12: 9216, 13: 10240, 14: 11264, 15: 12288,
    16: 13312, 17: 14336, 18: 15360, 19: 16384, 20: 17408,
    21: 18432, 22: 19456, 23: 20480, 24: 21504, 25: 22528
}
matrix_size = matrix_sizes.get(matrix_size_option, 512)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Memory check
memory_ok, message = check_memory()
if not memory_ok:
    if rank == 0:
        print(create_ascii_box("PROGRAM TERMINATION", [message]))
    sys.exit()

if rank == 0:
    print(create_ascii_box("STARTING MATRIX MULTIPLICATION", [message]))

# Divide workload among ranks
rows_per_process = matrix_size // size
start_row = rank * rows_per_process
end_row = (rank + 1) * rows_per_process - 1

# Create random matrices A and B (root only)
A = None
B = None
if rank == 0:
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)

# Scatter rows of A among processes
A_local = np.zeros((rows_per_process, matrix_size))
comm.Scatterv([A, rows_per_process * matrix_size, MPI.DOUBLE], A_local, root=0)

# Broadcast matrix B to all processes
B = comm.bcast(B, root=0)

# Perform local matrix multiplication
start_time = time.time()
C_local = np.dot(A_local, B)
computation_time = time.time() - start_time

# Gather computation times and results
computation_times = comm.gather(computation_time, root=0)
if rank == 0:
    # Combine partial results
    C = np.vstack(comm.gather(C_local, root=0))
else:
    comm.gather(C_local, root=0)

# Calculate performance metrics
if rank == 0:
    total_operations = 2 * (matrix_size ** 3)
    total_time = max(computation_times)
    flops = total_operations / total_time
    tops = flops / 1e12
    gflops = flops / 1e9
    avg_gflops_per_rank = gflops / size

    # Format computation times
    formatted_times = format_times(computation_times)

    # Print process responsibilities
    process_responsibilities = [
        f"[Rank {i}] Responsible for rows {i * rows_per_process} to {(i + 1) * rows_per_process - 1}"
        for i in range(size)
    ]
    print(create_ascii_box("PROCESS RESPONSIBILITIES", process_responsibilities))

    # Print matrix multiplication summary
    summary = [
        f"Matrix size: {matrix_size} x {matrix_size}",
        f"Number of processes: {size}",
        f"Total computation time (max across nodes): {total_time:.4f} seconds",
        f"Final result matrix shape: {C.shape}",
        f"Total Floating-Point Operations: {total_operations:,} operations",
        f"Achieved Performance: {tops:.4f} TOPS",
        f"Achieved Performance: {gflops:.2f} GFLOPS"
    ]
    print(create_ascii_box("MATRIX MULTIPLICATION COMPLETED", summary))

    # Print per rank data
    rank_data = [
        f"Operations per rank: {total_operations // size:,}",
        f"FLOPS per rank: {int(flops // size):,}",
        f"Average GFLOPS per rank: {avg_gflops_per_rank:.2f} GFLOPS",
        f"Time per operation: {total_time / total_operations:.10e} seconds"
    ]
    print(create_ascii_box("PER RANK DATA", rank_data))
