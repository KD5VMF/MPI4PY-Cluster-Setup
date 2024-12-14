import os
import subprocess

# Log function with improved formatting
def log(message, level="INFO", next_message=False):
    """Log messages with levels, separated by a blank line after pairs."""
    levels = {"INFO": "\033[94m[INFO]\033[0m", "SUCCESS": "\033[92m[SUCCESS]\033[0m", "ERROR": "\033[91m[ERROR]\033[0m"}
    print(f"{levels.get(level, '[INFO]')} {message}")
    if not next_message:  # Add a blank line after every two messages
        print()

# Function to run commands locally
def run_local_command(command, success_msg=None, error_msg=None):
    log(f"Running locally: {command}", next_message=True)
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log(success_msg if success_msg else f"{command} succeeded", "SUCCESS")
    except subprocess.CalledProcessError as e:
        log(error_msg if error_msg else f"{command} failed\n{e.stderr}", "ERROR")

# Function to run commands on a remote node via SSH
def run_remote_command(node, command, success_msg=None, error_msg=None):
    log(f"Running on {node}: {command}", next_message=True)
    ssh_command = f"ssh sysop@{node} '{command}'"
    try:
        result = subprocess.run(ssh_command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log(success_msg if success_msg else f"{command} succeeded on {node}", "SUCCESS")
    except subprocess.CalledProcessError as e:
        log(error_msg if error_msg else f"{command} failed on {node}\n{e.stderr}", "ERROR")

# Generate an SSH key if requested
def generate_ssh_key():
    log("Checking for existing SSH keys.", next_message=True)
    if not os.path.exists(os.path.expanduser("~/.ssh/id_rsa")):
        log("No SSH key found. Generating new key pair.", next_message=True)
        try:
            subprocess.run("ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa", shell=True, check=True)
            log("SSH key pair generated successfully.", "SUCCESS")
        except subprocess.CalledProcessError as e:
            log("Failed to generate SSH key pair.", "ERROR")
    else:
        log("Existing SSH key pair found.", "SUCCESS")

# Copy SSH keys to worker nodes
def copy_ssh_key_to_node(node):
    log(f"Copying SSH key to {node}.", next_message=True)
    try:
        subprocess.run(f"ssh-copy-id -o StrictHostKeyChecking=no sysop@{node}", shell=True, check=True)
        log(f"SSH key copied to {node} successfully.", "SUCCESS")
    except subprocess.CalledProcessError as e:
        log(f"Failed to copy SSH key to {node}.", "ERROR")

# Create the MPI host file
def create_host_file(start_ip, worker_count):
    log("Creating MPI host file.", next_message=True)
    try:
        with open(os.path.expanduser("~/mpi_hosts"), "w") as host_file:
            ip_parts = start_ip.split('.')
            base_ip = '.'.join(ip_parts[:3])
            last_octet = int(ip_parts[3])
            for i in range(worker_count):
                current_ip = f"{base_ip}.{last_octet + i}"
                host_file.write(f"{current_ip} slots=4\n")
        log("MPI host file created successfully.", "SUCCESS")
    except Exception as e:
        log(f"Failed to create host file: {str(e)}", "ERROR")

# Main setup function
def main():
    # Ask for environment name
    env_name = input("Enter the virtual environment name to use (default: envMPI): ").strip() or "envMPI"

    # Ask for the starting IP address
    start_ip = input("Enter the starting IP address (e.g., 192.168.0.191): ").strip()

    # Ask to setup coordinator node
    setup_coordinator = input("Do you want to setup the coordinator node? (yes/no): ").strip().lower() == "yes"
    if setup_coordinator:
        log("Setting up the coordinator node.", next_message=True)
        run_local_command("sudo apt update && sudo apt upgrade -y", "System updated successfully.", "System update failed.")
        run_local_command("sudo apt install -y python3 python3-venv python3-pip openmpi-bin openmpi-common libopenmpi-dev",
                          "MPI and Python development libraries installed successfully.",
                          "Failed to install MPI and Python development libraries.")
        run_local_command(f"python3 -m venv ~/{env_name}", f"Virtual environment {env_name} created.", f"Failed to create virtual environment {env_name}.")
        run_local_command(f"~/{env_name}/bin/pip install --upgrade pip setuptools wheel",
                          "Pip, setuptools, and wheel upgraded successfully.",
                          "Failed to upgrade pip, setuptools, and wheel.")
        run_local_command(f"~/{env_name}/bin/pip install mpi4py numpy scipy pandas matplotlib seaborn scikit-learn tensorflow tqdm",
                          "Python libraries for scientific computation installed successfully.",
                          "Failed to install Python libraries for scientific computation.")

    # Ask to setup worker nodes
    setup_workers = input("Do you want to setup worker nodes? (yes/no): ").strip().lower() == "yes"
    if setup_workers:
        worker_count = int(input("How many worker nodes do you want to setup? (max 24): ").strip())
        if worker_count > 0:
            ip_parts = start_ip.split('.')
            base_ip = '.'.join(ip_parts[:3])
            last_octet = int(ip_parts[3])
            for i in range(worker_count):
                current_ip = f"{base_ip}.{last_octet + i}"
                log(f"Setting up worker node {current_ip}.", next_message=True)
                run_remote_command(current_ip, "sudo apt update && sudo apt upgrade -y", "System updated successfully.", "System update failed.")
                run_remote_command(current_ip, f"python3 -m venv ~/{env_name}", f"Virtual environment {env_name} created on {current_ip}.", f"Failed to create virtual environment {env_name} on {current_ip}.")
            create_host_file(start_ip, worker_count)

    # Ask to setup SSH keys
    setup_ssh = input("Do you want to setup SSH keys for the coordinator and worker nodes? (yes/no): ").strip().lower() == "yes"
    if setup_ssh:
        generate_ssh_key()
        for i in range(worker_count):
            current_ip = f"{base_ip}.{last_octet + i}"
            copy_ssh_key_to_node(current_ip)

if __name__ == "__main__":
    main()
