from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix size options
matrix_sizes = {
    1: 256,
    2: 512,
    3: 1024,
    4: 2048,
    5: 4096,
    6: 4608,
    7: 5120,
    8: 6144,
    9: 6656,
    10: 7168,
}
matrix_option = 2  # Default matrix size
matrix_size = matrix_sizes.get(matrix_option, 512)

if rank == 0:
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)
else:
    A = None
    B = None

# Scatter rows of A to all processes
rows_per_process = matrix_size // size
A_local = np.zeros((rows_per_process, matrix_size))
comm.Scatter(A, A_local, root=0)

# Broadcast matrix B to all processes
B = comm.bcast(B, root=0)

# Local computation
C_local = np.dot(A_local, B)

# Gather results
C = None
if rank == 0:
    C = np.empty((matrix_size, matrix_size))
comm.Gather(C_local, C, root=0)

if rank == 0:
    print(f"Matrix multiplication complete. Resultant matrix shape: {C.shape}")
