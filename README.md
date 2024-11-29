
# Raspberry Pi MPI Cluster Setup

Welcome to the **Raspberry Pi MPI Cluster Setup** guide. This project provides a streamlined Python program to help users configure a fully functional MPI cluster using Raspberry Pi devices. The program automates the entire process, making it easy for beginners and experienced users alike.

## Features

- **Coordinator and Worker Node Setup**: Automatically configures the Raspberry Pi acting as the coordinator and adds up to 24 worker nodes.
- **Automated Environment Management**: Sets up a virtual Python environment (`envMPI`) on all nodes, installs necessary dependencies, and ensures compatibility with `mpi4py`.
- **Hostfile Generation**: Automatically creates an MPI-compatible `hostfile` with slots configuration for worker nodes.
- **SSH Key Management**: Optionally generates SSH keys for seamless communication between the coordinator and worker nodes.
- **Performance Testing**: Includes two example programs:
  - **Matrix Multiplication Benchmark**: Tests MPI-based distributed matrix multiplication.
  - **Password Cracker**: Demonstrates parallel processing by distributing password-cracking tasks across nodes.

## How It Works

### 1. Setup MPI System

Run the `setup_mpi_system.py` program to configure your cluster. The program prompts the user for input to guide the process and performs the following steps:

1. **Set Up the Coordinator Node**
    - Updates the system.
    - Installs OpenMPI and Python packages.
    - Creates a virtual environment (`envMPI`) and installs necessary Python libraries.

2. **Set Up Worker Nodes**
    - Adds up to 24 worker nodes based on the user’s configuration.
    - Automatically configures the `hostfile` with slots=4 for each node.

3. **SSH Key Management**
    - Optionally generates new SSH keys and distributes them to worker nodes.

4. **Verify Setup**
    - Confirms successful installation of `mpi4py` and OpenMPI on all nodes.

### 2. Run Example Programs

Two example programs are provided to test your cluster:

#### **Matrix Multiplication Benchmark**
This program performs distributed matrix multiplication using MPI to measure the performance of your cluster.

- Run with:
  ```bash
  mpirun --hostfile ~/mpi_hosts -np <number_of_processes> ~/envMPI/bin/python3 ~/matrix_multiplication.py <matrix_size_option>
  ```
  Replace `<number_of_processes>` with the total number of processes to use and `<matrix_size_option>` with a number between 1 and 40 to select the matrix size.

#### **Password Cracker**
A parallel password-cracking program that demonstrates distributed processing.

- Run with:
  ```bash
  mpirun --hostfile ~/mpi_hosts -np <number_of_processes> ~/envMPI/bin/python3 ~/password_cracker.py
  ```

## Installation Steps

1. Clone this repository:
    ```bash
    git clone <repository_link>
    ```

2. Navigate to the project directory:
    ```bash
    cd mpi_cluster_setup
    ```

3. Run the setup program:
    ```bash
    python3 setup_mpi_system.py
    ```

## Requirements

- **Hardware**:
  - Raspberry Pi 5 (Coordinator Node).
  - DeskPi Super6C cluster with up to 6 CM4 modules (Worker Nodes).

- **Software**:
  - Raspberry Pi OS Lite (64-bit) for all nodes.

- **Network**:
  - All nodes must be connected to the same local network.

## Example Output

Below is an example of the matrix multiplication program running on a cluster:

```
======================================================================
MPI Setup Test: Communication Across All Nodes
======================================================================
Hello from Rank 0 on node1
Hello from Rank 1 on node1
Hello from Rank 2 on node1
Hello from Rank 3 on node1
Hello from Rank 4 on node2
Hello from Rank 5 on node2
======================================================================
Total MPI processes: 6
MPI test completed successfully!
======================================================================
```

## Credits

This setup, programs, and documentation were created by **ChatGPT 4.0 in collaboration with Adam Figueroa**, based on the above DeskPi Super6C and Raspberry Pi 5 cluster configuration.

---

For further assistance, please open an issue or contact the maintainers of this repository.
