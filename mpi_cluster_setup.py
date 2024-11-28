
import os
import subprocess

# Ask the user for the virtual environment name
env_name = input("Enter the virtual environment name to use (default: envName): ").strip() or "envName"

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

# Function to verify the installation of mpi4py and environment
def verify_installation_on_node(node):
    command = f"ssh sysop@{node} '~/{env_name}/bin/python3 -c \'import mpi4py; print(mpi4py.__version__)\''"
    print(f"Verifying mpi4py installation on {node}")
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print(f"[VERIFIED] {node}: mpi4py version: {result.stdout.strip()}")
    else:
        print(f"[ERROR] {node}: Unable to verify mpi4py\n{result.stderr}")

# Main program
def main():
    print(f"Setting up worker nodes with virtual environment: {env_name}")
    for node in nodes:
        print(f"\n--- Setting up {node} ---")
        run_commands_on_node(node)
        verify_installation_on_node(node)

if __name__ == "__main__":
    main()
