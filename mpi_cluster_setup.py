import os
import subprocess

# Ask the user for the virtual environment name
env_name = input("Enter the virtual environment name to use (default: envMPI): ").strip() or "envMPI"

# Define your nodes
nodes = [
    "192.168.0.191",  # Worker node 1
    "192.168.0.192",  # Worker node 2
    "192.168.0.193",  # Worker node 3
    "192.168.0.194",  # Worker node 4
    "192.168.0.195",  # Worker node 5
    "192.168.0.196",  # Worker node 6
]

# Define the commands to run on each node
commands = [
    "sudo apt update && sudo apt upgrade -y",  # Update and upgrade system packages
    "sudo apt install -y python3 python3-venv python3-pip mpich libmpich-dev",  # Install required system packages
    f"python3 -m venv ~/{env_name}",  # Create virtual environment
    f"~/{env_name}/bin/pip install --upgrade pip setuptools wheel",  # Upgrade pip and build tools
    # Install additional Python libraries
    f"~/{env_name}/bin/pip install mpi4py numpy scipy pandas matplotlib seaborn scikit-learn tensorflow tqdm",
    f"~/{env_name}/bin/pip install pillow requests flask fastapi sqlalchemy psycopg2-binary opencv-python-headless sympy h5py boto3",
    f"~/{env_name}/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
    f"~/{env_name}/bin/pip install pycryptodome",  # Install pycryptodome for AES encryption
]

# Function to execute a command on a node via SSH
def run_command_on_node(node, command):
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

# Function to verify mpi4py and pycryptodome installation
def verify_installation(node):
    command = (
        f"ssh sysop@{node} '~/envMPI/bin/python3 -c "
        f"\"import mpi4py; from Crypto.Cipher import AES; print('mpi4py and pycryptodome verified!')\"'"
    )
    print(f"Verifying installation on {node}")
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print(f"[VERIFIED] {node}: {result.stdout.strip()}")
    else:
        print(f"[ERROR] {node}: Unable to verify installation\n{result.stderr}")

# Main program
def main():
    print(f"Setting up nodes with virtual environment: {env_name}")
    for node in nodes:
        print(f"\n--- Setting up {node} ---")
        for command in commands:
            run_command_on_node(node, command)
        verify_installation(node)

if __name__ == "__main__":
    main()
