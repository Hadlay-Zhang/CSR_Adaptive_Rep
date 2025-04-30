import argparse
import os
import math
from huggingface_hub import HfApi, upload_file
import sys

local_file_path = "train_500_0.50_90.ffcv" 
repo_id = "HadlayZ/ImageNet-1K-ffcv"       
repo_type = "dataset"                    
chunk_size_gb = 20                      
chunk_prefix = "train_chunk_"           
target_subdir = None                   
buffer_size = 10 * 1024 * 1024         

resume_from_chunk = 8 # set to the chunk index to resume from if uploading interrupted

chunk_size_bytes = chunk_size_gb * 1024 * 1024 * 1024
api = HfApi()

if not os.path.exists(local_file_path):
    print(f"Error: Input file not found at {local_file_path}")
    sys.exit(1)

file_size = os.path.getsize(local_file_path)
num_chunks = math.ceil(file_size / chunk_size_bytes)
chunk_digits = max(2, len(str(num_chunks - 1))) # Ensure at least 2 digits (00, 01...)

print(f"Total size: {file_size / (1024**3):.2f} GB")
print(f"Chunk size: {chunk_size_gb} GB")
print(f"Total number of chunks: {num_chunks}")

if resume_from_chunk >= num_chunks:
    print(f"Error: Resume chunk index {resume_from_chunk} is out of bounds (total chunks: {num_chunks}). Already finished?")
    sys.exit(1)
if resume_from_chunk < 0:
    print("Error: Resume chunk index cannot be negative.")
    sys.exit(1)


start_offset = resume_from_chunk * chunk_size_bytes # calculate the byte offset to start from

print(f"\nAttempting to resume from chunk {resume_from_chunk} / {num_chunks}")
print(f"Seeking to byte offset {start_offset} in {local_file_path}")

try:
    with open(local_file_path, "rb") as infile:
        infile.seek(start_offset)

        for i in range(resume_from_chunk, num_chunks):
            chunk_filename = f"{chunk_prefix}{i:0{chunk_digits}d}"
            # Use a temporary name while writing
            temp_chunk_path = f"{chunk_filename}.tmp"
            bytes_written = 0

            if os.path.exists(temp_chunk_path):
                print(f"Removing leftover temporary file: {temp_chunk_path}")
                os.remove(temp_chunk_path)
            if os.path.exists(chunk_filename):
                 print(f"Warning: Found existing chunk file {chunk_filename}. Will overwrite and re-upload.")
                 os.remove(chunk_filename)


            print(f"\nGenerating chunk {i+1}/{num_chunks}: {chunk_filename}")

            try:
                with open(temp_chunk_path, "wb") as outfile:
                    while bytes_written < chunk_size_bytes:
                        chunk = infile.read(buffer_size)
                        if not chunk:
                            break # End of input file
                        outfile.write(chunk)
                        bytes_written += len(chunk)

                if bytes_written > 0:
                     # Rename after successful write
                    os.rename(temp_chunk_path, chunk_filename)

                    print(f"Uploading {chunk_filename} ({bytes_written / (1024**2):.2f} MB)...")
                    path_in_repo = f"{target_subdir}/{chunk_filename}" if target_subdir else chunk_filename

                    upload_file(
                        path_or_fileobj=chunk_filename,
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        commit_message=f"Upload chunk {i+1}/{num_chunks}: {chunk_filename}"
                        # token=os.getenv("HF_TOKEN") # Use token if needed
                    )
                    print(f"Upload successful. Deleting local chunk: {chunk_filename}")
                    os.remove(chunk_filename)
                else:
                    if os.path.exists(temp_chunk_path):
                         os.remove(temp_chunk_path)
                    print(f"Skipping potentially empty chunk {i+1}/{num_chunks} at end of file.")


            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                if os.path.exists(temp_chunk_path):
                    os.remove(temp_chunk_path)
                if os.path.exists(chunk_filename): # If renamed before error
                    print(f"Removing potentially incomplete chunk file due to error: {chunk_filename}")
                    os.remove(chunk_filename)
                raise # Re-raise the exception to stop the process

            # If infile.read() returned nothing and we wrote nothing in this outer loop iteration
            if not chunk and bytes_written == 0:
                print("End of file reached.")
                break

    print("\nRemaining chunks processed and uploaded.")

except Exception as e:
    print(f"\nAn error occurred during the process: {e}")
except KeyboardInterrupt:
    print("\nProcess interrupted by user.")
    # Consider cleanup here if needed