import os
import glob
import shutil

SOURCE = "data/Pop1K7_piano_single"
OUTPUT = "data/pop100"

os.makedirs(OUTPUT, exist_ok=True)

files = sorted(glob.glob(SOURCE + "/*.mid"))[:100]

for f in files:
    shutil.copy2(f, OUTPUT)

print("COPIED:", len(files))
print("TO:", OUTPUT)