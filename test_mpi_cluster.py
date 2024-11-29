from mpi4py import MPI
import numpy as np
import time

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def latency_test():
    """Test round-trip latency between all nodes."""
    latencies = []
    for target in range(size):
        if rank == target:
            start_time = time.time()
            comm.send(None, dest=(rank + 1) % size, tag=11)
            comm.recv(source=(rank - 1) % size, tag=11)
            end_time = time.time()
            latencies.append((rank, end_time - start_time))
    gathered = comm.gather(latencies, root=0)
    if rank == 0:
        print("\nLatency Test Results:")
        for node_latencies in gathered:
            for latency in node_latencies:
                print(f"Node {latency[0]}: {latency[1]:.6f} seconds")

def bandwidth_test():
    """Test bandwidth by transferring a large array."""
    array_size = 10**7  # 10 million elements
    large_array = np.ones(array_size, dtype='float64')
    start_time = time.time()

    if rank == 0:
        comm.Send([large_array, MPI.DOUBLE], dest=(rank + 1) % size, tag=22)
        comm.Recv([large_array, MPI.DOUBLE], source=(rank - 1) % size, tag=22)
    else:
        comm.Recv([large_array, MPI.DOUBLE], source=(rank - 1) % size, tag=22)
        comm.Send([large_array, MPI.DOUBLE], dest=(rank + 1) % size, tag=22)

    end_time = time.time()
    transfer_time = end_time - start_time
    bandwidth = (large_array.nbytes / transfer_time) / (1024**2)  # MB/s
    gathered_bandwidth = comm.gather(bandwidth, root=0)
    if rank == 0:
        print("\nBandwidth Test Results:")
        for i, bw in enumerate(gathered_bandwidth):
            print(f"Node {i}: {bw:.2f} MB/s")

def compute_test():
    """Test computational performance via matrix multiplication."""
    matrix_size = 512
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)
    start_time = time.time()
    _ = np.dot(A, B)
    end_time = time.time()
    compute_time = end_time - start_time
    gathered_compute = comm.gather(compute_time, root=0)
    if rank == 0:
        print("\nCompute Test Results:")
        for i, ct in enumerate(gathered_compute):
            print(f"Node {i}: {ct:.6f} seconds for {matrix_size}x{matrix_size} matrix multiplication")

if rank == 0:
    print("======================================================================")
    print("MPI Cluster Performance Test")
    print("======================================================================")
comm.barrier()

latency_test()
comm.barrier()

bandwidth_test()
comm.barrier()

compute_test()
comm.barrier()

if rank == 0:
    print("======================================================================")
    print("MPI Cluster Test Completed Successfully!")
    print("======================================================================")
