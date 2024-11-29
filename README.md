
# Raspberry Pi MPI Cluster Management

Welcome to the Raspberry Pi MPI Cluster Management project! This repository contains tools and instructions to automate the setup and management of an MPI cluster, using Raspberry Pi devices as worker nodes and a Raspberry Pi 5 as the coordinator. Designed to streamline cluster deployment, these scripts will configure both the software and environment for scientific computation and distributed programming.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Cluster Specifications](#cluster-specifications)
3. [Programs Overview](#programs-overview)
   - [setup_mpi_system.py](#setup_mpi_systempy)
   - [matrix_multiplication.py](#matrix_multiplicationpy)
   - [password_cracker.py](#password_crackerpy)
4. [Getting Started](#getting-started)
5. [How to Push Files to Nodes](#how-to-push-files-to-nodes)
6. [Acknowledgments](#acknowledgments)

---

## Introduction

This project simplifies the setup of an MPI-based computation cluster with features such as:

- Automation of environment setup on the coordinator and worker nodes.
- MPI hostfile generation based on user-defined specifications.
- Streamlined deployment of computation programs across all nodes.

---

## Cluster Specifications

- **Coordinator Node:** Raspberry Pi 5 (512GB NVMe, 8GB RAM, static IP: `192.168.0.210`)
- **Worker Nodes:** Up to 24 Raspberry Pi Compute Modules 4 (256GB NVMe, 4GB RAM each)
- **Network:** All devices are connected via a DeskPi Super6C case with gigabit Ethernet.
- **Static IP Configuration:** 
  - `node1`: `192.168.0.191`
  - `node2`: `192.168.0.192`
  - `...`

---

## Programs Overview

### `setup_mpi_system.py`

**Purpose:** Automates the setup of the coordinator and worker nodes for MPI computation. The script:

1. Installs necessary libraries and dependencies.
2. Creates a Python virtual environment (default: `envMPI`).
3. Generates SSH keys for seamless communication (optional).
4. Creates an MPI hostfile based on user-defined worker node count.

**Key Features:**
- Intuitive prompts for user input.
- Detailed logging with progress indicators and error handling.

**Usage:**
Run the script on the coordinator node (e.g., Raspberry Pi 5):

```bash
python3 setup_mpi_system.py
```

Follow the prompts to configure the environment.

---

### `matrix_multiplication.py`

**Purpose:** Demonstrates matrix multiplication in a distributed MPI environment.

**How It Works:**
- The coordinator distributes matrices among nodes.
- Each node computes its share of the result matrix.
- The coordinator gathers and combines the results.

**Usage:**
Run the program across all nodes:

```bash
mpirun --hostfile ~/mpi_hosts -np 6 ~/envMPI/bin/python3 ~/matrix_multiplication.py 2
```

Here, `2` selects a 512x512 matrix for multiplication.

---

### `password_cracker.py`

**Purpose:** Demonstrates brute force password cracking with MPI. Useful for benchmarking and distributed task execution.

**How It Works:**
- The task is divided among nodes, each testing a subset of the keyspace.
- MPI is used for communication and synchronization.

**Usage:**
Run the program across all nodes:

```bash
mpirun --hostfile ~/mpi_hosts -np 6 ~/envMPI/bin/python3 ~/password_cracker.py
```

---

## Getting Started

1. **Set Static IPs:** Ensure all nodes have static IPs in the range `192.168.0.191` to `192.168.0.210`.
2. **Run `setup_mpi_system.py`:**
   ```bash
   python3 setup_mpi_system.py
   ```
   Follow the prompts to configure the coordinator and worker nodes.
3. **Push Files to Nodes:**
   Use `scp` to transfer `matrix_multiplication.py` and `password_cracker.py` to all nodes.
   ```bash
   scp ~/matrix_multiplication.py ~/password_cracker.py sysop@<node-ip>:/home/sysop/
   ```
4. **Verify MPI Installation:**
   Test the setup with a simple script like `test_mpi.py`.

---

## How to Push Files to Nodes

To push a file (e.g., `password_cracker.py`) to all worker nodes:

```bash
for i in {191..196}; do
  scp ~/password_cracker.py sysop@192.168.0.$i:/home/sysop/
done
```

---

## Acknowledgments

This project was collaboratively developed with the assistance of [ChatGPT](https://openai.com/), ensuring efficient automation and cluster management.

---

**Happy Computing!** 🎉
