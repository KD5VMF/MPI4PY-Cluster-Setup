
# 🚀 **Raspberry Pi Cluster - Setup and Program Guide**

Welcome to the comprehensive guide for setting up your Raspberry Pi cluster and running distributed programs! 
This guide includes:
- Step-by-step setup instructions for the cluster and virtual environment.
- Details about each program provided (`setup_mpi_system.py`, `password_cracker.py`, `matrix_multiplication.py`).
- Insights into the Raspberry Pi 5 setup and cluster hardware specifications.

---

## **1. Hardware Setup**

### **Raspberry Pi 5 Coordinator Node**
- **Case**: DeskPi Pro Case with active cooling.
- **Specifications**:
  - **CPU**: Quad-core ARM Cortex-A76 @ 2.4 GHz.
  - **RAM**: 8 GB LPDDR4X.
  - **Storage**: 512 GB NVMe SSD.
  - **Network**: Gigabit Ethernet for high-speed communication.
- **Role**: Acts as the main "coordinator" or master node for MPI jobs.

### **Raspberry Pi Compute Module 4 Worker Nodes**
- **Number of Nodes**: 6 CM4 modules.
- **Specifications**:
  - **CPU**: Quad-core ARM Cortex-A72 @ 1.5 GHz.
  - **RAM**: 4 GB per node.
  - **Storage**: 64 GB microSD cards (32 GB eMMC for `node1`).
- **Network**: All nodes connected via DeskPi Super6C Cluster Board.

### **Networking**
- All nodes communicate over a dedicated Gigabit Ethernet network with a static IP configuration:
  - **Coordinator Node**: 192.168.0.210.
  - **Worker Nodes**: 192.168.0.191 to 192.168.0.196.

---

## **2. Setting Up the Cluster**

### ✏️ **Script: `setup_mpi_system.py`**
This script automates the setup process for the coordinator and worker nodes, including installing MPI, Python libraries, and setting up SSH for password-less communication.

#### **Usage Instructions**
1. Run the script on the **coordinator node**:
   ```bash
   python3 setup_mpi_system.py
   ```
2. Follow the interactive prompts:
   - **Setup the coordinator node**: Installs Python, OpenMPI, and scientific libraries.
   - **Setup worker nodes**: Installs required packages remotely via SSH.
   - **Generate and distribute SSH keys**: Ensures seamless SSH access to all nodes.

#### **Features**
- **Automated Package Installation**: Python, OpenMPI, and libraries like `mpi4py` and `numpy`.
- **SSH Key Distribution**: Sets up password-less SSH access to all nodes.
- **MPI Host File Creation**: Generates the `~/mpi_hosts` file based on user input.

---

## **3. Programs for the Cluster**

### 🔑 **Program: `password_cracker.py`**
This program demonstrates distributed computing by cracking a simple password using MPI.

#### **How It Works**
- **Target**: Finds a single-character password from a known MD5 hash.
- **MPI Distribution**:
  - The password space is divided among the nodes.
  - Each node hashes its subset of characters and compares against the target hash.

#### **Usage**
Run the program with:
```bash
mpirun --hostfile ~/mpi_hosts -np <num_processes> python3 ~/mpi/password_cracker.py
```
- Replace `<num_processes>` with the total number of MPI processes.
- Example:
  ```bash
  mpirun --hostfile ~/mpi_hosts -np 6 python3 ~/mpi/password_cracker.py
  ```

---

### 🧮 **Program: `matrix_multiplication.py`**
This program performs distributed matrix multiplication using MPI.

#### **How It Works**
- **Input**: Two random matrices are generated on the root node (default size: 512x512).
- **MPI Distribution**:
  - Rows of the first matrix are scattered across all nodes.
  - The second matrix is broadcasted to all nodes.
- **Local Computation**: Each node calculates its portion of the resultant matrix.
- **Output**: The root node gathers the results and combines them.

#### **Usage**
Run the program with:
```bash
mpirun --hostfile ~/mpi_hosts -np <num_processes> python3 ~/mpi/matrix_multiplication.py
```
- Example:
  ```bash
  mpirun --hostfile ~/mpi_hosts -np 12 python3 ~/mpi/matrix_multiplication.py
  ```

---

## **4. Example MPI Workflow**

### **Editing and Running Scripts**
1. Edit the script:
   ```bash
   nano ~/mpi/<script_name>.py
   ```
2. Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).
3. Run the script using `mpirun`:
   ```bash
   mpirun --hostfile ~/mpi_hosts -np <num_processes> python3 ~/mpi/<script_name>.py
   ```

---

## **5. Example Output**

### **Password Cracker**
```plaintext
[INFO] Hello from Rank 0
[INFO] Hello from Rank 1
[INFO] Password found: a
```

### **Matrix Multiplication**
```plaintext
Matrix multiplication complete. Resultant matrix shape: (512, 512)
```

---

## 🙌 **Credits**
This setup, documentation, and programs were collaboratively developed using:
- Raspberry Pi 5 with DeskPi Pro Case.
- DeskPi Super6C Cluster Board and Raspberry Pi Compute Modules.
- MPI programs (`password_cracker.py`, `matrix_multiplication.py`) for demonstration.

---
