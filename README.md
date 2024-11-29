
# MPI4PY Cluster Setup and Example Programs

This repository includes the necessary setup program, example programs for distributed computing, and detailed instructions for working with MPI4PY on a Raspberry Pi cluster.

## Table of Contents
- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Example Programs](#example-programs)
- [Usage](#usage)

## Overview

This repository supports:
1. Setting up the coordinator node and worker nodes in a Raspberry Pi cluster.
2. Generating SSH keys for secure communication.
3. Creating and distributing an `mpi_hosts` file for MPI execution.
4. Installing required Python packages, including `mpi4py`, for distributed computation.

## Setup Instructions

### Prerequisites

- All Raspberry Pi nodes should have an operating system installed and networked.
- The main coordinator node (e.g., `PI5`) should be accessible via SSH to all worker nodes.

### Running the Setup Program

1. Place `setup_cluster.py` on the coordinator node.
2. Execute the program:
   ```bash
   python3 setup_cluster.py
   ```
3. Follow the prompts to:
   - Set up the coordinator node.
   - Add worker nodes by specifying their IP addresses and names.
   - Generate and distribute SSH keys.
   - Configure the environment (default: `envMPI`).

### Host File Format

The generated `mpi_hosts` file will look like:
```
192.168.0.191 slots=4
192.168.0.192 slots=4
192.168.0.193 slots=4
...
```

### Verifying Installation

Run the test program `test_mpi.py` to ensure that the MPI environment is correctly configured:
```bash
mpirun --hostfile ~/mpi_hosts -np 6 ~/envMPI/bin/python3 ~/test_mpi.py
```

## Example Programs

### 1. Matrix Multiplication

The program `matrix_multiplication.py` performs distributed matrix multiplication using MPI4PY.

Run the program with:
```bash
mpirun --hostfile ~/mpi_hosts -np <number_of_processes> ~/envMPI/bin/python3 matrix_multiplication.py <matrix_size_option>
```

#### Matrix Size Options:
- Option 1: 256x256
- Option 2: 512x512
- ...
- Option 10: 7168x7168

### 2. Password Cracker

The program `password_cracker.py` demonstrates a distributed brute-force password-cracking example.

Run the program with:
```bash
mpirun --hostfile ~/mpi_hosts -np <number_of_processes> ~/envMPI/bin/python3 password_cracker.py
```

## Usage

Refer to the program prompts for each example. Detailed performance metrics and node responsibilities are displayed upon execution.

---

This setup, programs, and documentation were created collaboratively by ChatGPT 4.0 and Adam Figueroa.
