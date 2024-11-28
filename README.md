
# 🖥️ MPI Distributed Computing Programs

This project showcases two distributed computing programs leveraging **MPI (Message Passing Interface)**. Designed for parallel execution across a multi-node cluster, these programs demonstrate:
1. **Password Cracking with Hashing and AES Encryption**
2. **Matrix Multiplication for Performance Benchmarking**

---

## 📋 Cluster and Master Node Setup

### **Master Node**
- **Device**: Raspberry Pi 5
- **Case**: Argon ONE V3 M.2 NVME PCIE Case
- **Specifications**:
  - 8GB RAM
  - 512GB NVMe storage

### **Cluster (Worker Nodes)**
- **6 Raspberry Pi Compute Modules 4 (CM4)**:
  - 4GB RAM each.
  - **Node1**: 32GB eMMC storage.
  - **Nodes 2-6**: 256GB NVMe storage.
- Connected via DeskPi Super6C carrier board.

### Access
The Raspberry Pi 5 is accessed via **Windows 11 CMD SSH** from the user's PC, which acts as the interface to run programs on the cluster.

---

## 🚀 Programs Overview

### 1. 🔒 **Password Cracker**
- **Features**:
  - Hash cracking using MD5 and SHA-256.
  - AES-128 and AES-256 encryption and decryption with timing metrics.
  - Interactive user interface for operation selection.
- **Purpose**:
  - Demonstrates MPI workload distribution for password cracking or encryption tasks.

### 2. 📊 **Matrix Multiplication**
- **Features**:
  - Distributed dense matrix multiplication.
  - Supports matrix sizes from 256x256 to 22,528x22,528.
  - Calculates GFLOPS and TOPS for performance benchmarking.
- **Purpose**:
  - Benchmarks computational performance across a distributed cluster.

---

## 🖥️ Requirements

### Cluster Setup
1. Ensure all nodes are accessible via SSH without a password:
   ```bash
   ssh-keygen -t rsa
   ssh-copy-id sysop@NODE_IP
   ```

2. Install necessary dependencies on all nodes:
   - MPI Environment:
     ```bash
     sudo apt update && sudo apt install -y mpich
     ```
   - Python Libraries:
     ```bash
     pip install mpi4py numpy
     ```

3. Prepare a hostfile for MPI execution (e.g., `~/mpi_hosts`):
   ```plaintext
   192.168.0.191 slots=4  # Node 1
   192.168.0.192 slots=4  # Node 2
   192.168.0.193 slots=4  # Node 3
   192.168.0.194 slots=4  # Node 4
   192.168.0.195 slots=4  # Node 5
   192.168.0.196 slots=4  # Node 6
   ```

---

## 📂 Deploying Programs to Worker Nodes

1. Copy both programs to all worker nodes:
   ```bash
   for NODE in 192.168.0.191 192.168.0.192 192.168.0.193 192.168.0.194 192.168.0.195 192.168.0.196; do
       scp password_cracker.py matrix_multiplication.py sysop@$NODE:~/
   done
   ```

2. Verify the files exist on each node:
   ```bash
   ssh sysop@192.168.0.191 ls ~/password_cracker.py ~/matrix_multiplication.py
   ```

---

## 🖥️ Running the Programs

### Password Cracker
#### Single Node Execution:
```bash
mpirun -np 4 python3 password_cracker.py
```

#### Multi-Node Execution:
```bash
mpirun --hostfile ~/mpi_hosts -np 24 python3 password_cracker.py
```

### Matrix Multiplication
#### Single Node Execution:
```bash
mpirun -np 4 python3 matrix_multiplication.py 2
```
- Replace `2` with the matrix size option (see matrix size table below).

#### Multi-Node Execution:
```bash
mpirun --hostfile ~/mpi_hosts -np 24 python3 matrix_multiplication.py 5
```
- Replace `5` with the matrix size option.

---

## 📏 Matrix Size Options

| Option | Matrix Size |
|--------|-------------|
| 1      | 256 x 256   |
| 2      | 512 x 512   |
| 3      | 1024 x 1024 |
| 4      | 2048 x 2048 |
| ...    | ...         |
| 40     | 22,528 x 22,528 |

---

## 🎯 Example Outputs

### Password Cracker
#### MD5 Hash Cracking:
```plaintext
Password Cracker Program
-------------------------
Select an algorithm:
1. MD5 (hashing)
2. SHA-256 (hashing)
...
Password found: secret
Total time taken: 0.3456 seconds
```

#### AES Encryption/Decryption:
```plaintext
AES Encryption Mode Selected
Key (hex): abc123...
Ciphertext (hex): def456...
Decrypted Password: secret
Encryption Time: 0.001234 seconds
```

### Matrix Multiplication
```plaintext
+----------------------------------------------------------------------+
|                      MATRIX MULTIPLICATION COMPLETED                |
+----------------------------------------------------------------------+
| Matrix size: 1024 x 1024                                             |
| Total computation time: 2.3456 seconds                               |
| Achieved Performance: 1.25 TOPS                                      |
| Average GFLOPS per rank: 50.00 GFLOPS                                |
+----------------------------------------------------------------------+
```

---

## ⚡ Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
