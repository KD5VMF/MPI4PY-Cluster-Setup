
import os
import subprocess

# Ask the user for the virtual environment name
env_name = input("Enter the virtual environment name to use (default: envMPI): ").strip() or "envMPI"

# Define your nodes
nodes = [
    "192.168.0.191",
    "192.168.0.192",
    "192.168.0.193",
    "192.168.0.194",
    "192.168.0.195",
    "192.168.0.196"
]

# Commands to set up each node
commands = [
    "sudo apt update && sudo apt upgrade -y",
    "sudo apt install -y python3 python3-venv python3-pip openmpi-bin openmpi-common libopenmpi-dev",
    f"python3 -m venv ~/{env_name}",
    f"~/{env_name}/bin/pip install --upgrade pip setuptools wheel",
    f"~/{env_name}/bin/pip install mpi4py numpy scipy pandas matplotlib seaborn scikit-learn tensorflow tqdm",
    f"~/{env_name}/bin/pip install pillow requests flask fastapi sqlalchemy psycopg2-binary opencv-python-headless sympy h5py boto3",
    f"~/{env_name}/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
    "echo 'export PATH=$PATH:/usr/local/bin:/usr/bin' >> ~/.bashrc",
    "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc",
    "source ~/.bashrc"
]

# Function to execute commands on a node via SSH
def run_commands_on_node(node):
    for command in commands:
        print(f"Running on {node}: {command}")
        result = subprocess.run(
            ["ssh", f"sysop@{node}", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            print(f"[SUCCESS] {node}: {command}")
        else:
            print(f"[ERROR] {node}: {command}\n{result.stderr}")

# Main program
def main():
    print(f"Setting up worker nodes with virtual environment: {env_name}")
    for node in nodes:
        print(f"\n--- Setting up {node} ---")
        run_commands_on_node(node)

if __name__ == "__main__":
    main()
