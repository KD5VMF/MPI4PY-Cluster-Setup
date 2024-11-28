
import os
import subprocess

# Ask the user for the virtual environment name
env_name = input("Enter the virtual environment name to use (default: envMPI: ").strip() or "envMPI"

# Define the commands to run on the Coordinator Node
commands = [
    "sudo apt update && sudo apt upgrade -y",  # Update and upgrade system packages
    "sudo apt install -y python3 python3-venv python3-pip mpich",  # Install required system packages
    f"python3 -m venv ~/{env_name}",  # Create virtual environment
    f"~/{env_name}/bin/pip install --upgrade pip setuptools wheel",  # Upgrade pip and build tools
    # Install additional Python libraries
    f"~/{env_name}/bin/pip install mpi4py numpy scipy pandas matplotlib seaborn scikit-learn tensorflow tqdm",
    f"~/{env_name}/bin/pip install pillow requests flask fastapi sqlalchemy psycopg2-binary opencv-python-headless sympy h5py boto3",
    # Install PyTorch CPU-only
    f"~/{env_name}/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
    # Ensure proper PATH and LD_LIBRARY_PATH are set
    "echo 'export PATH=$PATH:/usr/local/bin:/usr/bin' >> ~/.bashrc",
    "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc",
    "source ~/.bashrc",  # Reload bashrc
    # Ensure /tmp permissions are correct
    "sudo chmod 1777 /tmp",
]

# Function to execute a command locally
def run_command(command):
    print(f"Running command: {command}")
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
        print(f"[ERROR] {command}
{result.stderr}")

# Function to verify the installation
def verify_installation():
    command = f"~/{env_name}/bin/python3 -c 'import mpi4py, numpy, scipy, pandas, matplotlib, tensorflow, torch; print(mpi4py.__version__, numpy.__version__, scipy.__version__, pandas.__version__, matplotlib.__version__, tensorflow.__version__, torch.__version__)'"
    print(f"Verifying installation...")
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print(f"[VERIFIED]: mpi4py, numpy, scipy, pandas, matplotlib, tensorflow, torch versions: {result.stdout.strip()}")
    else:
        print(f"[ERROR]: Unable to verify installation
{result.stderr}")

# Main program
def main():
    print(f"Setting up the Coordinator Node with virtual environment: {env_name}")
    for command in commands:
        run_command(command)
    verify_installation()

if __name__ == "__main__":
    main()
