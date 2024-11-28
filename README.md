
# 🔒 Password Cracker with MPI and AES

This project demonstrates a distributed password cracking program using hashing and encryption algorithms. The program utilizes **MPI (Message Passing Interface)** for parallel processing across multiple nodes and includes support for **MD5**, **SHA-256**, and **AES encryption/decryption**.

---

## 🚀 Features

- **Hash Cracking**:
  - Supports **MD5** and **SHA-256** hashing algorithms.
  - Uses brute-force methods to match a given hash to its plaintext password.
  - Customizable character sets and password lengths.

- **AES Encryption/Decryption**:
  - Supports **AES-128** and **AES-256** encryption modes.
  - Uses proper padding and handles decryption errors gracefully.
  - Measures encryption and decryption times for benchmarking.

- **MPI Integration**:
  - Distributes brute-force workloads across multiple processes or nodes.
  - Gathers results efficiently to identify cracked passwords.

- **Interactive Interface**:
  - Allows users to choose between hashing or encryption modes.
  - Supports restarting or exiting the program after each operation.

---

## 📋 Requirements

### **Cluster Setup**
1. Ensure all nodes are accessible via SSH without a password.
   - Generate an SSH key (if not already created) on your control node:
     ```bash
     ssh-keygen -t rsa
     ```
   - Copy the key to each worker node (replace `NODE_IP` with your worker's IP):
     ```bash
     ssh-copy-id sysop@NODE_IP
     ```

2. Install the following dependencies on all nodes:
   - **MPI**: Install MPICH or OpenMPI on all nodes:
     ```bash
     sudo apt update && sudo apt install -y mpich
     ```
   - **Python**: Ensure Python 3.6+ is installed on all nodes.

3. Ensure your worker nodes are listed in an MPI hostfile, e.g., `~/mpi_hosts`:
   ```plaintext
   192.168.0.191 slots=4  # Worker node 1
   192.168.0.192 slots=4  # Worker node 2
   192.168.0.193 slots=4  # Worker node 3
   ...
   ```

---

## 🖥️ Usage

### 1. Push the Program to All Worker Nodes
1. Copy the `password_cracker.py` program to all worker nodes:
   ```bash
   for NODE in 192.168.0.191 192.168.0.192 192.168.0.193; do
       scp password_cracker.py sysop@$NODE:~/
   done
   ```

2. Verify the program exists on each node:
   ```bash
   ssh sysop@192.168.0.191 ls ~/password_cracker.py
   ```

### 2. Run the Program
- **On a Single Machine**:
  ```bash
  mpirun -np 4 python3 password_cracker.py
  ```

- **On a Cluster**:
  ```bash
  mpirun --hostfile ~/mpi_hosts -np 16 python3 password_cracker.py
  ```
  Replace `16` with the total number of processes across all nodes.

---

## 📂 Features in Detail

### **Hash Cracking**
- Choose from **MD5** or **SHA-256**.
- Brute-force cracking options:
  - **Generate a hash from a password and crack it.**
  - **Crack a given hash.**

### **AES Encryption/Decryption**
- Select **AES-128** or **AES-256** encryption modes.
- Generates random keys and initialization vectors (IVs).
- Provides encrypted ciphertext and decrypted plaintext with timing details.

### **MPI Workload Distribution**
- Distributes brute-force workload across all ranks (nodes/processes).
- Displays rank-specific workload distribution and performance details.

---

## 🎯 Example Output

### MD5 Hash Cracking
```plaintext
Password Cracker Program
-------------------------
Select an algorithm:
1. MD5 (hashing)
2. SHA-256 (hashing)
3. AES-128 (encryption)
4. AES-256 (encryption)
Enter the number for the algorithm (1-4): 1

1. Enter a password to hash and crack it automatically.
2. Crack an already known hash.
Enter 1 or 2: 1

Enter the password: test
Generated hash (md5): 098f6bcd4621d373cade4e832627b4f6
Max password length set to: 4
...
+-------------------------------------------------+
|                  CRACKING COMPLETE              |
+-------------------------------------------------+
| Password found: test
| Password length: 4
| Found by rank: 2
+-------------------------------------------------+
| Total time taken: 0.3456 seconds
| Total processes: 4
+-------------------------------------------------+
```

### AES Encryption/Decryption
```plaintext
AES Encryption Mode Selected
Enter the password to encrypt: secret

Key (hex): e3f1a1a2b3c4d5e6f7g8h9i0j1k2l3m4
IV (hex): a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
Ciphertext (hex): d3e4f5g6h7i8j9k0l1m2n3o4p5q6r7s8

Decrypted Password: secret

+-------------------------------------------------+
|               AES OPERATION COMPLETE            |
+-------------------------------------------------+
| Encryption Time: 0.001234 seconds
| Decryption Time: 0.001056 seconds
+-------------------------------------------------+
```

---

## ⚡ Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or feedback, feel free to reach out via GitHub issues.

---
