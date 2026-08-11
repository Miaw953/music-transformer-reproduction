import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import MusicTransformer
from masking import create_look_ahead_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


sequence_lengths = [128, 256, 512, 1024]

model = MusicTransformer().to(device)
model.eval()

print("Model loaded.")
print("Sequence lengths:", sequence_lengths)
print()


for seq_len in sequence_lengths:
    print(f"Testing sequence length: {seq_len}")

    x = torch.randint(
        low=0,
        high=model.vocab_size,
        size=(1, seq_len),
        device=device
    )

    mask = create_look_ahead_mask(seq_len).to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(x, mask=mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
        extra_memory = peak_memory - start_memory
        extra_memory_mb = extra_memory / (1024 ** 2)

        print(f"Extra peak GPU memory: {extra_memory_mb:.2f} MB")

    print(f"Forward time: {elapsed_time:.4f} seconds")
    print()

    del x, mask, output