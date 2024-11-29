
# 🎯 **Raspberry Pi Cluster MPI Guide**

### 📋 **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Editing Your MPI Script](#editing-your-mpi-script)
3. [Running the MPI Program](#running-the-mpi-program)
4. [Example Output](#example-output)
5. [Credits](#credits)

---

### 🛠️ **Prerequisites**
Before running your MPI program:
- Ensure that all Raspberry Pi nodes have `mpi4py` installed and are on the same local network.
- Configure your `~/mpi_hosts` file with the IP addresses or hostnames of your nodes.

Example `~/mpi_hosts`:
```text
192.168.0.191 slots=4  # Node 1
192.168.0.192 slots=4  # Node 2
192.168.0.193 slots=4  # Node 3
192.168.0.194 slots=4  # Node 4
192.168.0.195 slots=4  # Node 5
192.168.0.196 slots=4  # Node 6
```

---

### ✏️ **Editing Your MPI Script**
1. Open your script with `nano`:
   ```bash
   nano ~/mpi/<script_name>.py
   ```
   Replace `<script_name>` with the name of your script (e.g., `matrix_multiplication.py`).

2. Modify the script as needed.

3. Save the script:
   - Press `Ctrl + O` to save changes.
   - Press `Enter` to confirm.
   - Press `Ctrl + X` to exit.

---

### 🚀 **Running the MPI Program**
Run your program across the cluster using `mpirun`:
```bash
mpirun --hostfile ~/mpi_hosts -np <num_processes> python3 ~/mpi/<script_name>.py
```
- Replace `<num_processes>` with the total number of processes (e.g., 24 for 6 nodes with 4 slots each).
- Replace `<script_name>` with the name of your Python script.

Example:
```bash
mpirun --hostfile ~/mpi_hosts -np 24 python3 ~/mpi/matrix_multiplication.py
```

---

### 📜 **Example Output**
Below is an example output from a test MPI program running on the cluster:

```plaintext
==================================================
🎉 MPI Setup Test: Communication Across All Nodes 🎉
==================================================
Hello from Rank 0 on node1
Hello from Rank 1 on node1
Hello from Rank 2 on node1
Hello from Rank 3 on node1
Hello from Rank 4 on node2
Hello from Rank 5 on node2
==================================================
✅ Total MPI processes: 6
✅ MPI test completed successfully!
==================================================
```

---

### 🎨 **Make It Fancy**
Want to jazz up your output? Try these tricks:

#### **Add ASCII Art with `pyfiglet`**
Install `pyfiglet`:
```bash
pip install pyfiglet
```

Use it in your script:
```python
import pyfiglet

ascii_banner = pyfiglet.figlet_format("MPI Test")
print(ascii_banner)
```

#### **Add Colors**
Use ANSI escape codes to colorize your output:
```python
print("[1;32mHello from MPI Cluster![0m")  # Green text
print("[1;31mError: Something went wrong[0m")  # Red text
```

---

### 🙌 **Credits**
This guide, scripts, and documentation were created with ❤️ by **[Your Name]** and **ChatGPT 4.0**, inspired by the power of distributed computing on Raspberry Pi clusters.

---
