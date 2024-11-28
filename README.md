
# 🖥️ MPI Distributed Computing Programs and Cluster Setup

This project includes tools and programs to set up and utilize a Raspberry Pi-based MPI cluster for distributed computing. The repository demonstrates the following:

1. **Cluster Setup with `mpi_cluster_setup.py`**
   - Automates the setup of Python environments and dependencies across cluster nodes.

2. **Distributed Computing Programs**
   - 🔒 **Password Cracker**: Hash cracking with MD5, SHA-256, and AES encryption.
   - 📊 **Matrix Multiplication**: Performance benchmarking using large matrix operations.

---

## 📋 Cluster and Coordinator Node Setup

### **Cluster Setup in Detail**

This setup is based on the author's current configuration and serves as a guide for new users. 

### **Coordinator Node**
- **Device**: Raspberry Pi 5
- **Case**: Argon ONE V3 M.2 NVME PCIE Case
- **Specifications**:
  - 8GB RAM
  - 512GB NVMe storage
- Accessed via **Windows 11 CMD SSH**.

### **Cluster (Worker Nodes)**
- **6 Raspberry Pi Compute Modules 4 (CM4)**:
  - 4GB RAM each.
  - **Node1**: 32GB eMMC storage.
  - **Nodes 2-6**: 256GB NVMe storage.
- Connected via DeskPi Super6C carrier board.

---

## 🚀 Full Setup Guide (New Cluster)

### **1. Preparing the Host File**
The host file lists the IPs of all nodes and their configuration for MPI:
1. Create the file on the Coordinator Node (Raspberry Pi 5):
   ```bash
   nano ~/mpi_hosts
   ```
2. Add the following content (adjust IPs as needed):
   ```plaintext
   192.168.0.191 slots=4  # Node 1
   192.168.0.192 slots=4  # Node 2
   192.168.0.193 slots=4  # Node 3
   192.168.0.194 slots=4  # Node 4
   192.168.0.195 slots=4  # Node 5
   192.168.0.196 slots=4  # Node 6
   ```
3. Save and exit Nano (`Ctrl + O`, `Enter`, `Ctrl + X`).

### **2. Generating SSH Keys on the Coordinator Node**
1. Generate an SSH key (if not already created):
   ```bash
   ssh-keygen -t rsa -b 2048
   ```
2. Press `Enter` to save the key in the default location (`~/.ssh/id_rsa`), and optionally set a passphrase.

### **3. Copying SSH Keys to Worker Nodes**
1. Use the following command to copy the key to each worker node:
   ```bash
   for NODE in 192.168.0.191 192.168.0.192 192.168.0.193 192.168.0.194 192.168.0.195 192.168.0.196; do
       ssh-copy-id sysop@$NODE
   done
   ```
2. Verify that passwordless SSH works:
   ```bash
   ssh sysop@192.168.0.191
   ```

### **4. Running `mpi_cluster_setup.py`**
The `mpi_cluster_setup.py` script simplifies the setup of Python and MPI across nodes.
1. Copy the script to the Coordinator Node:
   ```bash
   scp mpi_cluster_setup.py sysop@192.168.0.191:~/
   ```
2. Run the script on the Coordinator Node:
   ```bash
   python3 mpi_cluster_setup.py
   ```

---

## 📂 Deploying Programs to Worker Nodes

1. Copy the programs (`password_cracker.py`, `matrix_multiplication.py`) to all worker nodes:
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
|                      MATRIX MULTIPLICATION COMPLETED                 |
+----------------------------------------------------------------------+
| Matrix size: 1024 x 1024                                             |
| Total computation time: 2.3456 seconds                               |
| Achieved Performance: 1.25 TOPS                                      |
| Average GFLOPS per rank: 50.00 GFLOPS                                |
+----------------------------------------------------------------------+
```

---

## 🛠️ Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✒️ Attribution
This setup, programs, and documentation were created by ChatGPT 4.0 in collaboration with Adam Figueroa, based on the current DeskPi Super6C and Raspberry Pi 5 cluster configuration.
