
# MPI Cluster Utilities

Welcome to the **MPI Cluster Utilities**! This repository is designed to assist you in managing and testing a Raspberry Pi cluster with ease. With this toolkit, you can set up, manage, and test your cluster, as well as perform computational tasks like matrix multiplication and password cracking.

---

## Included Programs

### 1. **setup_mpi_system.py**
This script automates the setup of your MPI cluster. It helps in configuring the coordinator node (main node) and worker nodes (CM4 nodes) with minimal user intervention.

- **Features:**
  - Creates a Python virtual environment (`envMPI` by default).
  - Installs MPI and Python dependencies (like mpi4py, numpy, pandas, etc.).
  - Configures the host file (`mpi_hosts`) for your cluster.
  - Sets up SSH keys and distributes them across nodes.
  - Automates worker node setup.

- **Usage:**
  ```bash
  python3 setup_mpi_system.py
  ```

### 2. **test_mpi_cluster.py**
This program thoroughly tests your cluster's performance, including:
- **Latency Tests:** Measures communication delays between nodes.
- **Bandwidth Tests:** Tests data transfer rates.
- **Compute Tests:** Benchmarks 512x512 matrix multiplications.

- **Usage:**
  ```bash
  mpirun --hostfile ~/mpi_hosts -np 24 ~/envMPI/bin/python3 ~/test_mpi_cluster.py
  ```

### 3. **matrix_multiplication.py**
This program performs distributed matrix multiplication using MPI. It benchmarks the cluster's compute power for matrix sizes ranging from small to very large.

- **Features:**
  - User-selectable matrix size (1-25 options).
  - Reports detailed benchmarks, including FLOPS, GFLOPS, and per-process performance.

- **Usage:**
  ```bash
  mpirun --hostfile ~/mpi_hosts -np 24 ~/envMPI/bin/python3 ~/matrix_multiplication.py
  ```

### 4. **password_cracker.py**
This is an MPI-based password cracker that can crack hashed passwords using a distributed brute-force approach.

- **Features:**
  - Asks the user for the target password to crack.
  - Automatically detects the password length and distributes the workload across nodes.
  - Reports detailed cracking benchmarks.

- **Usage:**
  ```bash
  mpirun --hostfile ~/mpi_hosts -np 24 ~/envMPI/bin/python3 ~/password_cracker.py
  ```

---

## How to Push Programs to Worker Nodes

To distribute a program (e.g., `matrix_multiplication.py`) to all worker nodes:

1. Use the `scp` command to copy the program to all nodes:
   ```bash
   for i in {192..196}; do scp ~/matrix_multiplication.py sysop@192.168.0.$i:~/; done
   ```

2. Ensure the program has the correct permissions:
   ```bash
   chmod +x ~/matrix_multiplication.py
   ```

---

## Your Cluster Setup

### Hardware
- **Coordinator Node: Raspberry Pi 5**
  - **Case:** DeskPi Super6C Case
  - **Storage:** 512GB NVMe
- **Worker Nodes: 6 CM4 Modules**
  - **Storage:** 256GB NVMe per module
  - **Network Configuration:** Static IPs in the 192.168.0.191-196 range.

### Software
- Raspberry Pi OS Lite 64-bit on all nodes.
- Python 3.11 with `mpi4py` and other dependencies installed in a virtual environment (`envMPI`).

---

## Example Commands

1. **Run the matrix multiplication test:**
   ```bash
   mpirun --hostfile ~/mpi_hosts -np 24 ~/envMPI/bin/python3 ~/matrix_multiplication.py
   ```

2. **Run the password cracker:**
   ```bash
   mpirun --hostfile ~/mpi_hosts -np 24 ~/envMPI/bin/python3 ~/password_cracker.py
   ```

3. **Test cluster performance:**
   ```bash
   mpirun --hostfile ~/mpi_hosts -np 24 ~/envMPI/bin/python3 ~/test_mpi_cluster.py
   ```

---

## Credit

This repository was created with the assistance of **ChatGPT**. It combines automation, benchmarking, and computational utilities tailored for Raspberry Pi clusters.

