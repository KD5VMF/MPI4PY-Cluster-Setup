from mpi4py import MPI
import hashlib

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

target_hash = hashlib.md5(b"password").hexdigest()
charset = "abcdefghijklmnopqrstuvwxyz"
chunk_size = len(charset) // size

start = rank * chunk_size
end = (rank + 1) * chunk_size
found = None

for char in charset[start:end]:
    if hashlib.md5(char.encode()).hexdigest() == target_hash:
        found = char
        break

found = comm.gather(found, root=0)

if rank == 0:
    result = next((item for item in found if item), None)
    if result:
        print(f"Password found: {result}")
    else:
        print("Password not found.")
