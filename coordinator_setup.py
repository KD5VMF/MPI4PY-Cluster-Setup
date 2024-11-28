import os
import subprocess

# Ask the user for the virtual environment name
env_name = input("Enter the virtual environment name to use (default: envName): ").strip() or "envName"

# Commands to set up the Coordinator Node
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

# Function to run commands locally
def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print(f"[SUCCESS] {command}")
    else:
        print(f"[ERROR] {command}\n{result.stderr}")

# Function to verify mpi4py installation and environment
def verify_installation():
    command = f"~/{env_name}/bin/python3 -c 'import mpi4py; print(mpi4py.__version__)'"
    print("Verifying mpi4py installation...")
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print(f"[VERIFIED] mpi4py version: {result.stdout.strip()}")
    else:
        print(f"[ERROR] Unable to verify mpi4py\n{result.stderr}")

# Main program
def main():
    print(f"Setting up Coordinator Node with virtual environment: {env_name}")
    for command in commands:
        run_command(command)
    verify_installation()

if __name__ == "__main__":
    main()
