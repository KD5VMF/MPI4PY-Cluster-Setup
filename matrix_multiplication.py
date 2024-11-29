from mpi4py import MPI
import itertools
import string
import hashlib
import time

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Password characters (you can extend this to include uppercase, symbols, etc.)
CHARACTERS = string.ascii_lowercase + string.digits

def hash_password(password):
    """Hash the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_passwords(length, start, end):
    """Generate passwords of a given length for this process."""
    all_passwords = itertools.product(CHARACTERS, repeat=length)
    return itertools.islice(all_passwords, start, end)

if rank == 0:
    print("="*70)
    print("Password Cracker (MPI) with User Input and Optimized Algorithm")
    print("="*70)
    target_password = input("Enter the password to crack: ").strip()
    target_hash = hash_password(target_password)
    print(f"[INFO] Target password hash: {target_hash}")
    print(f"[INFO] Starting password cracking...")
    print("="*70)
else:
    target_hash = None

# Broadcast the target hash to all processes
target_hash = comm.bcast(target_hash, root=0)

# Auto-detection loop
start_time = time.time()
password_found = None
password_length = 1

while password_found is None:
    if rank == 0:
        print(f"[INFO] Trying passwords of length {password_length}...")

    # Calculate workload distribution
    total_combinations = len(CHARACTERS) ** password_length
    chunk_size = total_combinations // size
    start_index = rank * chunk_size
    end_index = total_combinations if rank == size - 1 else start_index + chunk_size

    # Generate passwords for this process
    for password_tuple in generate_passwords(password_length, start_index, end_index):
        password = ''.join(password_tuple)
        if hash_password(password) == target_hash:
            password_found = password
            break

    # Gather results from all processes
    password_found = comm.allreduce(password_found if password_found else "", op=MPI.SUM)
    password_found = password_found if password_found else None

    if password_found is None:
        password_length += 1

end_time = time.time()

if rank == 0:
    print("="*70)
    if password_found:
        print(f"[SUCCESS] Password found: {password_found}")
        print(f"[INFO] Time taken: {end_time - start_time:.2f} seconds")
    else:
        print("[ERROR] Password not found.")
    print("="*70)
