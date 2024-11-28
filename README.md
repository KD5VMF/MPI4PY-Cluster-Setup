
# 🐍 MPI Cluster Python Setup Script

![MPI Cluster Setup](https://img.shields.io/badge/MPI-Cluster-brightgreen) ![Python](https://img.shields.io/badge/Python-3.x-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

This script automates the setup of Python virtual environments and essential libraries for a multi-node MPI cluster. Ideal for users working with `mpi4py`, data analysis, machine learning, or distributed computing.

---

## 🔧 Features

- **Automated Setup**:
  - Updates and upgrades system packages.
  - Installs required system dependencies (`python3`, `mpich`, etc.).
  - Sets up Python virtual environments on multiple nodes.
- **Library Installation**:
  - Installs essential Python libraries, including:
    - `mpi4py`, `numpy`, `scipy`, `pandas`, `tensorflow`, `torch` (with CUDA support), and more.
- **Easy Node Management**:
  - Works with any standard MPI host configuration.
- **Verification**:
  - Verifies successful installation of critical libraries.

---

## 🖥️ Requirements

- **Cluster Nodes**:
  - SSH access to all nodes with the same user credentials.
  - Python 3.6 or later installed on all nodes.
- **Control Node**:
  - Python 3.6+ with `subprocess` module available.
  - Access to all cluster nodes via SSH.

---

## 🚀 Usage

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your_username/mpi-cluster-setup.git
cd mpi-cluster-setup
```

### **Step 2: Edit the Script (Optional)**
Update the `nodes` list in `mpi_cluster_setup.py` to match your cluster's IPs or hostnames:
```python
nodes = [
    "NODE_IP_1",
    "NODE_IP_2",
    "NODE_IP_3",
    "NODE_IP_4",
    "NODE_IP_5",
    "NODE_IP_6"
]
```

### **Step 3: Run the Script**
Make the script executable and run it:
```bash
chmod +x mpi_cluster_setup.py
python3 mpi_cluster_setup.py
```

- **Prompt**: The script will ask for a virtual environment name (default: `envName`).
- The script sets up each node and installs the libraries automatically.

---

## 📦 Installed Libraries

### Core Libraries:
- `mpi4py`: MPI for Python.
- `numpy`, `scipy`, `pandas`: Numerical and data analysis.
- `matplotlib`, `seaborn`: Visualization.
- `tensorflow`, `torch` (with CUDA 12.1), `torchvision`, `torchaudio`: Machine learning and deep learning.
- `tqdm`: Progress bars.

### Utilities:
- `pillow`, `requests`, `flask`, `fastapi`
- `sqlalchemy`, `psycopg2-binary`, `h5py`, `boto3`

---

## 🛠️ Example Test
Run a program across your cluster with:
```bash
mpirun --hostfile ~/mpi_hosts -np 24 ~/envName/bin/python3 your_script.py
```

---

## 🎯 Contributing
Feel free to open issues or submit pull requests to improve this script. Contributions are welcome!

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
