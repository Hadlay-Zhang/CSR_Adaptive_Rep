import numpy as np
import faiss
import time
import pandas as pd
from argparse import ArgumentParser
from os import path, makedirs
import gc

parser=ArgumentParser()
parser.add_argument('--topk', default=8, type=int,help='the number of topk')
parser.add_argument('--gpus', default=0, type=int, help='Number of GPUs to use for search (0 for CPU)')
args = parser.parse_args()

topk = args.topk
num_gpus_to_use = args.gpus
root = f'./CSR_topk_{topk}/'
dataset = 'V1'
index_type = 'exactl2'
k = 2048 # shortlist length

db_csv = f'{dataset}_train_topk_{topk}'+ '-X.npy'
query_csv = f'{dataset}_val_topk_{topk}'+ '-X.npy'

# check GPU availability
ngpus_available = faiss.get_num_gpus()
print(f"Number of GPUs available: {ngpus_available}")
if num_gpus_to_use > ngpus_available:
    raise ValueError(f"Requested {num_gpus_to_use} GPUs, but only {ngpus_available} are available.")
elif num_gpus_to_use > 0:
    print(f"Will use {num_gpus_to_use} GPU(s) for search.")
else:
    print("Will use CPU for search.")


start_load_query = time.time()
queryset = np.load(root+query_csv)
print(f"Queryset load time= {time.time() - start_load_query:.3f} sec")


index_dir = root + 'index_files/'
neighbors_out_dir = root + "neighbors/" # output dir path
if not path.isdir(index_dir):
    print(f"Creating directory: {index_dir}")
    makedirs(index_dir)
if not path.isdir(neighbors_out_dir):
    print(f"Creating directory: {neighbors_out_dir}")
    makedirs(neighbors_out_dir)


index_file = path.join(index_dir, f"{dataset}_{index_type}.index")
cpu_index = None
if path.exists(index_file):
    print(f"Loading index file: {index_file}")
    start_load_index = time.time()
    cpu_index = faiss.read_index(index_file)
    if cpu_index is None or cpu_index.ntotal == 0:
         raise RuntimeError(f"Failed to load a valid index from {index_file}")
    print(f"CPU Index loaded (ntotal={cpu_index.ntotal}). Load time= {time.time() - start_load_index:.3f} sec")

else:
    print(f"Generating index file: {index_file}")
    db_filepath = root + db_csv
    start_load_db = time.time()
    xb = np.ascontiguousarray(np.load(db_filepath), dtype=np.float32)
    print(f"Database load time= {time.time() - start_load_db:.3f} sec")

    faiss.normalize_L2(xb)
    d = xb.shape[1]  # dimension
    nb = xb.shape[0]  # database size
    print(f"Database shape: {xb.shape}")

    start_build = time.time()
    if index_type == 'exactl2':
        print("Building CPU IndexFlatL2")
        cpu_index = faiss.IndexFlatL2(d)
    else:
        del xb
        gc.collect()
        raise NotImplementedError(f"Index building for CPU index type '{index_type}' is not implemented.")

    cpu_index.add(xb)
    print(f"Writing CPU index to disk: {index_file}")
    faiss.write_index(cpu_index, index_file)
    print(f"CPU Index build and write time= {time.time() - start_build:.3f} sec")
    del xb
    gc.collect()

# Queries
xq = np.ascontiguousarray(queryset, dtype=np.float32)
print("Normalizing query vectors (L2)...")
faiss.normalize_L2(xq)
nq = xq.shape[0]
print(f"Queries shape: {xq.shape}")

# GPU Acceleration
index = None
if num_gpus_to_use > 0:
    if cpu_index is None or cpu_index.ntotal == 0:
        raise RuntimeError("CPU index is not loaded or is empty")

    print(f"Attempting to copy CPU index (ntotal={cpu_index.ntotal}) to {num_gpus_to_use} GPU(s) using float16")
    start_clone = time.time()

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True       # shard index across GPUs (good for IndexFlat)
    co.useFloat16 = False  # float16 to potentially halve memory usage & speed up
    gpu_list = list(range(num_gpus_to_use))
    index = faiss.index_cpu_to_all_gpus(cpu_index)
    print(f"GPU index clone successful. Clone time= {time.time() - start_clone:.3f} sec")

else:
    print("Using CPU index for search.")
    if cpu_index is None:
        raise RuntimeError("CPU index not loaded or built, cannot proceed with CPU search.")
    index = cpu_index

if index is None:
    raise RuntimeError("Index object was not successfully created or assigned before search step.")
if index.ntotal == 0:
    raise RuntimeError("Index is empty, cannot perform search.")


print(f"Starting {k}-NN search for {nq} queries on {'GPU(s)' if num_gpus_to_use > 0 else 'CPU'}...")
start_search = time.time()
D, I = index.search(xq, k) # D=distances, I=indices (shape: nq x k)
end_search = time.time() - start_search

search_device = "GPU" if num_gpus_to_use > 0 else "CPU"
print(f"{search_device} {k}-NN search completed.")
print(f"Search time= {end_search:.6f} sec")

if end_search > 0:
    print(f"Throughput: {nq / end_search:.2f} queries/sec")
else:
    print("Search time was negligible (throughput calculation skipped).")

start_write = time.time()
nn_filename = f"{index_type}_{k}shortlist_{dataset}.csv"
nn_filepath = path.join(neighbors_out_dir, nn_filename)
print(f"Writing neighbor indices to: {nn_filepath}")
pd.DataFrame(I).to_csv(nn_filepath, header=None, index=None)
end_write = time.time() - start_write
print("NN file write time= %0.3f sec\n" % (end_write))