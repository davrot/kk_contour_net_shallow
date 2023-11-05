import glob
import os
import re
import shutil

"""
get performances from .pt files
"""

directory = "./trained_models"
string = "Natural"
final_path = "./trained_corners"


# list of all files in the directory
files = glob.glob(directory + "/*.pt")

# filter
filtered_files = [f for f in files if string in f]

# group by seed
seed_files = {}
for f in filtered_files:
    # get seed from filename
    match = re.search(r"_seed(\d+)_", f)
    if match:
        seed = int(match.group(1))
        if seed not in seed_files:
            seed_files[seed] = []
        seed_files[seed].append(f)


# get saved cnn largests epoch
newest_files = {}
for seed, files in seed_files.items():
    max_epoch = -1
    newest_file = None
    for f in files:
        # search for epoch
        match = re.search(r"_(\d+)Epoch_", f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                newest_file = f
    newest_files[seed] = newest_file

print(len(newest_files))

# move files to new folder
os.makedirs(final_path, exist_ok=True)

# Copy the files to the new folder
for seed, file in newest_files.items():
    shutil.copy(file, os.path.join(final_path, os.path.basename(file)))
