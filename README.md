
# 🖥️ MPI Distributed Computing Programs and Cluster Setup   - NOT READY YET / 11/28/2024

This repository includes tools, programs, and documentation to set up and utilize a Raspberry Pi-based MPI cluster for distributed computing. The repository demonstrates the following:

1. **Cluster Setup with `mpi_cluster_setup.py` (Workers) and `coordinator_setup.py` (Coordinator)**
   - Automates the setup of Python environments and dependencies across cluster nodes.

2. **Distributed Computing Programs**
   - 🔒 **Password Cracker**: Hash cracking with MD5, SHA-256, and AES encryption.
   - 📊 **Matrix Multiplication**: Performance benchmarking using large matrix operations.

---

## 📋 Cluster and Coordinator Node Setup

### **Cluster Overview**

### **Coordinator Node**
- **Device**: Raspberry Pi 5
- **Case**: Argon ONE V3 M.2 NVME PCIE Case
- **Specifications**:
  - 8GB RAM
  - 512GB NVMe storage
- **Purpose**: Sets up the cluster and manages execution of distributed programs.

### **Cluster (Worker Nodes)**
- **6 Raspberry Pi Compute Modules 4 (CM4)**:
  - 4GB RAM each.
  - **Node1**: 32GB eMMC storage.
  - **Nodes 2-6**: 256GB NVMe storage.
- Connected via DeskPi Super6C carrier board.

---

## 📏 Matrix Size Options

| Option | Matrix Size |
|--------|-------------|
| 1      | 256 x 256   |
| 2      | 512 x 512   |
| 3      | 1024 x 1024 |
| 4      | 2048 x 2048 |
| 5      | 4096 x 4096 |
| 6      | 4608 x 4608 |
| 7      | 5120 x 5120 |
| 8      | 6144 x 6144 |
| 9      | 6656 x 6656 |
| 10     | 7168 x 7168 |
| 11     | 7680 x 7680 |
| 12     | 8192 x 8192 |
| 13     | 8704 x 8704 |
| 14     | 9216 x 9216 |
| 15     | 9728 x 9728 |
| 16     | 10240 x 10240 |
| 17     | 10752 x 10752 |
| 18     | 11264 x 11264 |
| 19     | 11776 x 11776 |
| 20     | 12288 x 12288 |

### Memory Considerations
The cluster, with each node having 4GB of RAM, performs optimally for options up to **11 (7680 x 7680)**. If **option 12 (8192 x 8192)** or larger is selected, the memory requirements may exceed the available RAM per node. This can trigger the use of swap space, significantly slowing down the program's execution due to disk I/O bottlenecks.

For best performance, consider keeping matrix sizes within the available RAM limits of your nodes.

---

## 🚀 Running the Programs

### Password Cracker
#### Single Node Execution:
```bash
mpirun -np 4 python3 password_cracker.py
```
- Replace `4` with the number of processes on your single node.

#### Multi-Node Execution:
```bash
mpirun --hostfile ~/mpi_hosts -np 24 python3 password_cracker.py
```
- Replace `24` with the total number of processes across all nodes.

### Matrix Multiplication
#### Single Node Execution:
```bash
mpirun -np 4 python3 matrix_multiplication.py 2
```
- Replace `2` with the desired matrix size option (see matrix size table above).

#### Multi-Node Execution:
```bash
mpirun --hostfile ~/mpi_hosts -np 24 python3 matrix_multiplication.py 5
```
- Replace `5` with the desired matrix size option.

---

## 🚀 Setup Scripts

### 🛠️ Coordinator Node Setup (`coordinator_setup.py`)
The `coordinator_setup.py` script automates the setup of the Coordinator Node (Raspberry Pi 5) by installing the necessary tools, configuring Python environments, and preparing it to manage the worker nodes.

#### How to Use
1. **Copy `coordinator_setup.py` to the Coordinator Node**:
   ```bash
   scp coordinator_setup.py sysop@192.168.0.210:~/
   ```

2. **Run the script on the Coordinator Node**:
   ```bash
   python3 coordinator_setup.py
   ```

3. The script performs the following:
   - Updates and upgrades the system.
   - Installs MPI, Python libraries, and SSH utilities.
   - Creates a Python virtual environment and installs required libraries.
   - Verifies the installation of key Python libraries.

### 🛠️ Worker Node Setup (`mpi_cluster_setup.py`)
The `mpi_cluster_setup.py` script simplifies the setup of worker nodes in the cluster. It installs Python and MPI dependencies across multiple nodes.

#### How to Use
1. **Copy `mpi_cluster_setup.py` to the Coordinator Node**:
   ```bash
   scp mpi_cluster_setup.py sysop@192.168.0.210:~/
   ```

2. **Run the script on the Coordinator Node**:
   ```bash
   python3 mpi_cluster_setup.py
   ```

3. The script will:
   - Update and upgrade all worker nodes.
   - Install MPI, Python libraries, and create virtual environments.
   - Verify the installation on all worker nodes.

---

## 📂 Deploying Programs to Worker Nodes

1. Copy the distributed programs (`password_cracker.py`, `matrix_multiplication.py`) to all worker nodes:
   ```bash
   for NODE in 192.168.0.191 192.168.0.192 192.168.0.193 192.168.0.194 192.168.0.195 192.168.0.196; do
       scp password_cracker.py matrix_multiplication.py sysop@$NODE:~/
   done
   ```

2. Verify the files exist on each worker node:
   ```bash
   ssh sysop@192.168.0.191 ls ~/password_cracker.py ~/matrix_multiplication.py
   ```

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
|                      MATRIX MULTIPLICATION COMPLETED                 |
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

---

## ✒️ Attribution
This setup, programs, and documentation was created by ChatGPT 4o in collaboration with Adam Figueroa, based on the above DeskPi Super6C and Raspberry Pi 5 cluster configuration.
