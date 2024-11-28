import os
import subprocess

# Define your nodes
nodes = [
    "192.168.0.191",  # Worker node 1
    "192.168.0.192",  # Worker node 2
    "192.168.0.193",  # Worker node 3
    "192.168.0.194",  # Worker node 4
    "192.168.0.195",  # Worker node 5
    "192.168.0.196"   # Worker node 6
]

# Commands to clean up the system
cleanup_commands = [
    "rm -rf ~/envMPI",  # Remove the virtual environment
    "sudo apt purge -y python3-venv python3-pip mpich libmpich-dev openmpi-bin openmpi-common libopenmpi-dev",
    "sudo apt autoremove -y",
    "sudo apt clean",
    "find ~ -name '*.pyc' -delete",  # Remove compiled Python files
    "find ~ -name '__pycache__' -delete"  # Remove Python cache directories
]

# Function to execute cleanup commands on a remote node
def cleanup_node(node):
    for command in cleanup_commands:
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

# Function to clean up the coordinator node
def cleanup_coordinator():
    for command in cleanup_commands:
        print(f"Running on Coordinator: {command}")
        result = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            print(f"[SUCCESS] Coordinator: {command}")
        else:
            print(f"[ERROR] Coordinator: {command}\n{result.stderr}")

# Main program
def main():
    print("Starting cleanup on all nodes...")
    
    # Cleanup coordinator node
    print("\n--- Cleaning up Coordinator Node ---")
    cleanup_coordinator()
    
    # Cleanup worker nodes
    for node in nodes:
        print(f"\n--- Cleaning up {node} ---")
        cleanup_node(node)
    
    print("\nCleanup completed on all nodes.")

if __name__ == "__main__":
    main()
