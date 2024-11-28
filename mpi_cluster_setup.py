import os
import subprocess

# Ask the user for the virtual environment name
env_name = input("Enter the virtual environment name to use (default: envMPI): ").strip() or "envMPI"

# Commands to set up the coordinator node
setup_commands = [
    "sudo apt update && sudo apt upgrade -y",
    "sudo apt install -y python3 python3-venv python3-pip mpich libmpich-dev",
    f"python3 -m venv ~/{env_name}",
    f"~/{env_name}/bin/pip install --upgrade pip setuptools wheel",
    f"~/{env_name}/bin/pip install mpi4py numpy scipy pandas matplotlib seaborn scikit-learn tensorflow tqdm",
    f"~/{env_name}/bin/pip install pillow requests flask fastapi sqlalchemy psycopg2-binary opencv-python-headless sympy h5py boto3",
    f"~/{env_name}/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
    f"~/{env_name}/bin/pip install pycryptodome"
]

# Function to execute commands on the coordinator node
def setup_coordinator():
    for command in setup_commands:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"[SUCCESS] {command}")
        else:
            print(f"[ERROR] {command}\n{result.stderr}")

# Verify mpi4py installation
def verify_mpi4py():
    command = f"~/{env_name}/bin/python3 -c 'import mpi4py; print(mpi4py.__version__)'"
    print("Verifying mpi4py installation...")
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print(f"[VERIFIED] mpi4py version: {result.stdout.strip()}")
    else:
        print(f"[ERROR] Unable to verify mpi4py\n{result.stderr}")

def main():
    print(f"Setting up Coordinator Node with virtual environment: {env_name}")
    setup_coordinator()
    verify_mpi4py()
    print("\nCoordinator setup completed.")

if __name__ == "__main__":
    main()
