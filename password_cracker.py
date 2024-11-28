import hashlib
import itertools
import string
import sys
import time
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Clear terminal on start (only on master node)
if rank == 0:
    os.system('clear')

# Hashing and encryption functions
def hash_password(password, algorithm):
    """Hashes the password using MD5 or SHA-256."""
    if algorithm == "md5":
        return hashlib.md5(password.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(password.encode()).hexdigest()
    else:
        raise ValueError("Unsupported hash algorithm")

def aes_encrypt(password, key_size=128):
    """Encrypts the password using AES with proper padding."""
    key = get_random_bytes(key_size // 8)  # Generate a random AES key
    cipher = AES.new(key, AES.MODE_CBC)    # AES in CBC mode
    padded_password = pad(password.encode(), AES.block_size)  # Ensure correct padding
    ciphertext = cipher.encrypt(padded_password)
    return key, cipher.iv, ciphertext

def aes_decrypt(key, iv, ciphertext):
    """Decrypts the ciphertext and handles padding errors gracefully."""
    cipher = AES.new(key, AES.MODE_CBC, iv)  # Include the IV here
    try:
        decrypted_data = cipher.decrypt(ciphertext)
        plaintext = unpad(decrypted_data, AES.block_size)  # Unpad the plaintext
        return plaintext.decode()
    except ValueError as e:
        return f"Decryption failed: {str(e)}"

# Brute-force function
def brute_force(target_hash, char_set, max_length, algorithm):
    """Attempts to brute-force the password hash."""
    for length in range(1, max_length + 1):
        for i, candidate in enumerate(itertools.product(char_set, repeat=length)):
            if i % size != rank:
                continue
            candidate_str = ''.join(candidate)
            if hash_password(candidate_str, algorithm) == target_hash:
                return candidate_str, length, i
    return None, None, None

# Main logic encapsulated in a loop
if rank == 0:
    while True:
        print("\nPassword Cracker Program")
        print("-------------------------")
        print("Select an algorithm:")
        print("1. MD5 (hashing)")
        print("2. SHA-256 (hashing)")
        print("3. AES-128 (encryption)")
        print("4. AES-256 (encryption)")
        algo_choice = input("Enter the number for the algorithm (1-4): ").strip()

        # Map choice to algorithm
        algo_map = {
            "1": "md5",
            "2": "sha256",
            "3": "aes128",
            "4": "aes256"
        }
        algorithm = algo_map.get(algo_choice, None)

        if algorithm in ["md5", "sha256"]:
            print("\n1. Enter a password to hash and crack it automatically.")
            print("2. Crack an already known hash.")
            choice = input("Enter 1 or 2: ").strip()

            if choice == "1":
                password = input("Enter the password: ").strip()
                try:
                    target_hash = hash_password(password, algorithm)
                    max_length = len(password)
                    print(f"\nGenerated hash ({algorithm}): {target_hash}")
                    print(f"Max password length set to: {max_length}")
                except ValueError as e:
                    print(e)
                    sys.exit(1)
            elif choice == "2":
                target_hash = input("Enter the hash to crack: ").strip()
                try:
                    max_length = int(input("Enter max password length: "))
                except ValueError:
                    print("Invalid max password length. Please enter a number.")
                    sys.exit(1)
            else:
                print("Invalid choice.")
                sys.exit(1)

            char_set = string.ascii_letters + string.digits  # Modify as needed
            start_time = time.time()  # Start timer

        elif algorithm in ["aes128", "aes256"]:
            print("\nAES Encryption Mode Selected")
            password = input("Enter the password to encrypt: ").strip()
            key_size = 128 if algorithm == "aes128" else 256
            start_time = time.time()  # Start timer for AES operations
            key, iv, ciphertext = aes_encrypt(password, key_size)
            encryption_time = time.time() - start_time  # End timer for encryption
            print(f"\nKey (hex): {key.hex()}")
            print(f"IV (hex): {iv.hex()}")
            print(f"Ciphertext (hex): {ciphertext.hex()}")

            # Start timer for decryption
            start_time = time.time()
            decrypted_password = aes_decrypt(key, iv, ciphertext)
            decryption_time = time.time() - start_time  # End timer for decryption
            print("\nDecrypted Password:", decrypted_password)

            # Print time statistics for AES operations
            print("\n+-------------------------------------------------+")
            print("|               AES OPERATION COMPLETE            |")
            print("+-------------------------------------------------+")
            print(f"| Encryption Time: {encryption_time:.6f} seconds")
            print(f"| Decryption Time: {decryption_time:.6f} seconds")
            print("+-------------------------------------------------+")

            # After encryption/decryption, prompt to exit or restart
            while True:
                choice = input("\nDo you want to (E)xit or (R)estart? ").strip().lower()
                if choice == 'e':
                    print("Exiting program.")
                    sys.exit(0)
                elif choice == 'r':
                    break  # Restart the loop
                else:
                    print("Invalid choice. Please enter 'E' to exit or 'R' to restart.")
            continue  # Restart the main loop

        else:
            print("Invalid algorithm selection.")
            continue  # Restart the main loop

        # Broadcast inputs for cracking to all nodes
        target_hash = comm.bcast(target_hash, root=0)
        algorithm = comm.bcast(algorithm, root=0)
        max_length = comm.bcast(max_length, root=0)
        char_set = comm.bcast(char_set, root=0)
        start_time = comm.bcast(start_time, root=0)

        # Perform brute force (only for hashing algorithms)
        if algorithm in ["md5", "sha256"]:
            result, length, index = brute_force(target_hash, char_set, max_length, algorithm)
            results = comm.gather((result, length, index), root=0)

            if rank == 0:
                # Master node: determine the correct result
                end_time = time.time()  # End timer
                cracked_password = None
                for r in results:
                    if r[0]:
                        cracked_password, length, index = r
                        break

                # Print results
                print("\n+-------------------------------------------------+")
                print("|                  CRACKING COMPLETE              |")
                print("+-------------------------------------------------+")
                if cracked_password:
                    print(f"| Password found: {cracked_password}")
                    print(f"| Password length: {length}")
                    print(f"| Found by rank: {results.index((cracked_password, length, index))}")
                else:
                    print("| Password not found within specified parameters. |")
                print("+-------------------------------------------------+")
                print(f"| Total time taken: {end_time - start_time:.4f} seconds")
                print(f"| Total processes: {size}")
                print("+-------------------------------------------------+")

                # Workload details
                print("\nWorkload Distribution:")
                for rank_id, r in enumerate(results):
                    if r[0]:  # Only print details for ranks that found a match
                        print(f"- Rank {rank_id} processed passwords of length {r[1]} and found the match.")
                        break

                # After cracking, prompt to exit or restart
                while True:
                    choice = input("\nDo you want to (E)xit or (R)estart? ").strip().lower()
                    if choice == 'e':
                        print("Exiting program.")
                        sys.exit(0)
                    elif choice == 'r':
                        break  # Restart the loop
                    else:
                        print("Invalid choice. Please enter 'E' to exit or 'R' to restart.")
        else:
            # For AES modes, the program already handles exit/restart
            pass  # No action needed

else:
    # Worker nodes don't interact; wait for master to broadcast data
    while True:
        target_hash = comm.bcast(None, root=0)
        algorithm = comm.bcast(None, root=0)
        max_length = comm.bcast(None, root=0)
        char_set = comm.bcast(None, root=0)
        start_time = comm.bcast(None, root=0)

        # Perform brute force (only for hashing algorithms)
        if algorithm in ["md5", "sha256"]:
            result, length, index = brute_force(target_hash, char_set, max_length, algorithm)
            comm.gather((result, length, index), root=0)
        else:
            # For AES modes, worker nodes do nothing
            pass
