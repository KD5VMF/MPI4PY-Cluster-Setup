#!/usr/bin/env python3
import os
import sys
import ipaddress
import subprocess

# ---------- Logging ----------
def log(message, level="INFO", next_message=False):
    """Log messages with levels, separated by a blank line after pairs."""
    levels = {
        "INFO": "\033[94m[INFO]\033[0m",
        "SUCCESS": "\033[92m[SUCCESS]\033[0m",
        "ERROR": "\033[91m[ERROR]\033[0m",
    }
    print(f"{levels.get(level, '[INFO]')} {message}")
    if not next_message:
        print()

# ---------- Command helpers ----------
def run_local_command(command, success_msg=None, error_msg=None):
    log(f"Running locally: {command}", next_message=True)
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.stdout.strip():
            print(result.stdout)
        if result.stderr.strip():
            print(result.stderr)
        log(success_msg if success_msg else f"{command} succeeded", "SUCCESS")
    except subprocess.CalledProcessError as e:
        log(error_msg if error_msg else f"{command} failed\n{e.stderr}", "ERROR")

def run_remote_command(node, command, success_msg=None, error_msg=None, username="sysop"):
    log(f"Running on {node}: {command}", next_message=True)
    ssh_command = (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"{username}@{node} '{command}'"
    )
    try:
        result = subprocess.run(
            ssh_command, shell=True, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.stdout.strip():
            print(result.stdout)
        if result.stderr.strip():
            print(result.stderr)
        log(success_msg if success_msg else f"{command} succeeded on {node}", "SUCCESS")
    except subprocess.CalledProcessError as e:
        log(error_msg if error_msg else f"{command} failed on {node}\n{e.stderr}", "ERROR")

# ---------- SSH key management ----------
def generate_ssh_key():
    log("Checking for existing SSH keys.", next_message=True)
    home = os.path.expanduser("~")
    id_ed = os.path.join(home, ".ssh", "id_ed25519")
    id_rsa = os.path.join(home, ".ssh", "id_rsa")

    if not (os.path.exists(id_ed) or os.path.exists(id_rsa)):
        log("No SSH key found. Generating new ED25519 key pair.", next_message=True)
        try:
            subprocess.run("ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519", shell=True, check=True)
            log("SSH key pair generated successfully.", "SUCCESS")
        except subprocess.CalledProcessError:
            log("Failed to generate ED25519 key; falling back to RSA.", "ERROR", next_message=True)
            try:
                subprocess.run("ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa", shell=True, check=True)
                log("RSA SSH key pair generated successfully.", "SUCCESS")
            except subprocess.CalledProcessError as e:
                log(f"Failed to generate SSH key pair.\n{e}", "ERROR")
    else:
        which = "ED25519" if os.path.exists(id_ed) else "RSA"
        log(f"Existing SSH key pair found ({which}).", "SUCCESS")

def copy_ssh_key_to_node(node, username="sysop"):
    log(f"Copying SSH key to {node}.", next_message=True)
    try:
        subprocess.run(
            f"ssh-copy-id -o StrictHostKeyChecking=no {username}@{node}",
            shell=True, check=True
        )
        log(f"SSH key copied to {node} successfully.", "SUCCESS")
    except subprocess.CalledProcessError as e:
        log(f"Failed to copy SSH key to {node}.\n{e}", "ERROR")

# ---------- Hostfile ----------
def create_host_file(nodes, slots_per_node=4, filename="~/mpi_hosts"):
    path = os.path.expanduser(filename)
    log(f"Creating MPI host file at {path}.", next_message=True)
    try:
        with open(path, "w") as host_file:
            for idx, ip in enumerate(nodes, start=1):
                host_file.write(f"{ip} slots={slots_per_node}  # Worker node {idx}\n")
        log("MPI host file created successfully.", "SUCCESS")
    except Exception as e:
        log(f"Failed to create host file: {str(e)}", "ERROR")

# ---------- IP helpers ----------
def build_node_list(start_ip: str, count: int):
    try:
        base = ipaddress.ip_address(start_ip)
    except ValueError:
        raise ValueError(f"Invalid IP address: {start_ip}")
    if count < 1:
        raise ValueError("Worker count must be >= 1")
    nodes = []
    for i in range(count):
        next_ip = base + i
        if next_ip.packed[-1] == 0 or next_ip.packed[-1] == 255:
            raise ValueError("Resulting IP would be network/broadcast address; adjust the range.")
        nodes.append(str(next_ip))
    return nodes

# ---------- Setup steps ----------
def install_local_packages():
    run_local_command(
        "sudo apt-get update && sudo apt-get -y upgrade",
        "System updated successfully.",
        "System update failed."
    )
    run_local_command(
        "sudo apt-get install -y python3 python3-venv python3-pip "
        "build-essential openmpi-bin openmpi-common libopenmpi-dev",
        "MPI and Python development libraries installed successfully.",
        "Failed to install MPI and Python development libraries."
    )

def setup_local_env(env_name):
    run_local_command(
        f"python3 -m venv ~/{env_name}",
        f"Virtual environment {env_name} created.",
        f"Failed to create virtual environment {env_name}."
    )
    run_local_command(
        f"~/{env_name}/bin/pip install --upgrade pip setuptools wheel",
        "Pip, setuptools, and wheel upgraded successfully.",
        "Failed to upgrade pip, setuptools, and wheel."
    )
    # Keep deps slim and portable on ARM; TensorFlow is intentionally omitted here.
    run_local_command(
        f"~/{env_name}/bin/pip install mpi4py numpy scipy pandas matplotlib scikit-learn tqdm",
        "Scientific Python stack installed successfully.",
        "Failed to install Python libraries."
    )

def setup_worker(node, env_name, username="sysop"):
    # Packages: python+venv+pip, build-essential, and OpenMPI (headers + runtime)
    run_remote_command(
        node,
        "sudo apt-get update && sudo apt-get -y upgrade",
        "System updated successfully.",
        "System update failed.",
        username=username
    )
    run_remote_command(
        node,
        "sudo apt-get install -y python3 python3-venv python3-pip "
        "build-essential openmpi-bin openmpi-common libopenmpi-dev",
        "MPI and Python dev libs installed.",
        "Failed to install MPI/Python dev libs.",
        username=username
    )
    run_remote_command(
        node,
        f"python3 -m venv ~/{env_name}",
        f"Virtual environment {env_name} created.",
        f"Failed to create virtual environment {env_name}.",
        username=username
    )
    run_remote_command(
        node,
        f"~/{env_name}/bin/pip install --upgrade pip setuptools wheel",
        "Pip tooling upgraded.",
        "Failed to upgrade pip tooling.",
        username=username
    )
    run_remote_command(
        node,
        f"~/{env_name}/bin/pip install mpi4py numpy",
        "Installed mpi4py and numpy.",
        "Failed to install Python packages.",
        username=username
    )

# ---------- Main ----------
def main():
    print("\n=== Pi Cluster Bootstrap ===\n")
    username = (input("Remote username (default: sysop): ").strip() or "sysop")

    # Ask general topology first so later steps have the data regardless of choices
    env_name = input("Virtual environment name (default: envMPI): ").strip() or "envMPI"
    start_ip = input("Starting worker IP (e.g., 192.168.0.191): ").strip()
    try:
        default_count = 6
        count_str = input(f"How many worker nodes? (default: {default_count}): ").strip()
        worker_count = int(count_str) if count_str else default_count
        if worker_count < 1 or worker_count > 24:
            raise ValueError
    except ValueError:
        log("Worker count must be an integer between 1 and 24.", "ERROR")
        sys.exit(1)

    try:
        nodes = build_node_list(start_ip, worker_count)
    except ValueError as e:
        log(str(e), "ERROR")
        sys.exit(1)

    # Coordinator setup
    setup_coordinator = input("Set up the coordinator (this Pi 5)? (yes/no): ").strip().lower() == "yes"
    if setup_coordinator:
        log("Setting up the coordinator node.", next_message=True)
        install_local_packages()
        setup_local_env(env_name)

    # Worker setup
    setup_workers = input(f"Set up {worker_count} worker nodes now? (yes/no): ").strip().lower() == "yes"
    if setup_workers:
        for idx, node in enumerate(nodes, start=1):
            log(f"Setting up worker node {idx}/{worker_count} at {node}.", next_message=True)
            setup_worker(node, env_name, username=username)

    # Hostfile (always helpful)
    create_host_file(nodes, slots_per_node=4, filename="~/mpi_hosts")

    # SSH keys
    setup_ssh = input("Set up SSH keys (passwordless) to workers? (yes/no): ").strip().lower() == "yes"
    if setup_ssh:
        generate_ssh_key()
        for node in nodes:
            copy_ssh_key_to_node(node, username=username)

    # Final tips
    print("\n=== Next steps ===")
    print(f"1) Verify MPI run across nodes:")
    np = worker_count * 4
    print(f"   mpirun --hostfile ~/mpi_hosts -np {np} ~/{env_name}/bin/python3 - <<'PY'\n"
          "from mpi4py import MPI\n"
          "import socket\n"
          "comm=MPI.COMM_WORLD\n"
          "print(f'Hello from rank {comm.rank}/{comm.size} on {socket.gethostname()}')\n"
          "PY")
    print("\n2) If you need more Python packages for your workloads, install them both locally and on each worker venv.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
