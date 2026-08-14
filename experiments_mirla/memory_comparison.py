import os
import gc
import csv
import torch
import matplotlib.pyplot as plt

# Allow imports from project root
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from layers import skew


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One attention head dimension
D_HEAD = 64

# Increase sequence length progressively
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048]

os.makedirs("results", exist_ok=True)


def clear_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ---------------------------------------------------------
# NAIVE RELATIVE POSITION CALCULATION
# ---------------------------------------------------------

def naive_relative_attention_memory(L):
    """
    Naive implementation.

    Explicitly creates a relative-position embedding
    for EVERY query-key pair.

    Shape of relative tensor:
        [L, L, D_HEAD]

    This is the expensive intermediate that the
    Music Transformer paper avoids.
    """

    clear_gpu()

    q = torch.randn(L, D_HEAD, device=DEVICE)

    # Relative embedding table
    relative_table = torch.randn(L, D_HEAD, device=DEVICE)

    # Relative distances between every pair of positions
    positions = torch.arange(L, device=DEVICE)

    relative_indices = (
        positions[:, None] - positions[None, :]
    ).abs()

    relative_indices = torch.clamp(
        relative_indices,
        max=L - 1
    )

    # EXPENSIVE L x L x D tensor
    relative_tensor = relative_table[relative_indices]

    # Compute relative attention score
    scores = torch.einsum(
        "id,ijd->ij",
        q,
        relative_tensor
    )

    torch.cuda.synchronize()

    memory = peak_memory_mb()

    del q
    del relative_table
    del relative_indices
    del relative_tensor
    del scores

    clear_gpu()

    return memory


# ---------------------------------------------------------
# MEMORY-EFFICIENT MUSIC TRANSFORMER METHOD
# ---------------------------------------------------------

def efficient_relative_attention_memory(L):
    """
    Music Transformer style calculation.

    Instead of creating [L, L, D_HEAD],
    calculate Q @ E^T first.

    Intermediate:
        [L, L]

    Then skew() rearranges the scores.
    """

    clear_gpu()

    q = torch.randn(L, D_HEAD, device=DEVICE)

    relative_embeddings = torch.randn(
        L,
        D_HEAD,
        device=DEVICE
    )

    # Compact multiplication
    qe = torch.matmul(
        q,
        relative_embeddings.transpose(-1, -2)
    )

    # Add dimensions expected by skew:
    # [batch, head, L, L]
    qe = qe.unsqueeze(0).unsqueeze(0)

    scores = skew(qe)

    torch.cuda.synchronize()

    memory = peak_memory_mb()

    del q
    del relative_embeddings
    del qe
    del scores

    clear_gpu()

    return memory


# ---------------------------------------------------------
# RUN EXPERIMENT
# ---------------------------------------------------------

print("Device:", DEVICE)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "This experiment requires a CUDA GPU. Run it in Google Colab."
    )

results = []

for L in SEQUENCE_LENGTHS:

    print(f"\nTesting sequence length: {L}")

    try:
        naive_memory = naive_relative_attention_memory(L)
        print(f"Naive:     {naive_memory:.2f} MB")

    except torch.cuda.OutOfMemoryError:
        naive_memory = None
        print("Naive:     OUT OF MEMORY")
        clear_gpu()

    efficient_memory = efficient_relative_attention_memory(L)

    print(f"Efficient: {efficient_memory:.2f} MB")

    results.append(
        {
            "sequence_length": L,
            "naive_memory_mb": naive_memory,
            "efficient_memory_mb": efficient_memory,
        }
    )


# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

csv_path = "results/memory_comparison.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sequence_length",
            "naive_memory_mb",
            "efficient_memory_mb",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", csv_path)


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

lengths = [r["sequence_length"] for r in results]

naive = [
    r["naive_memory_mb"]
    for r in results
]

efficient = [
    r["efficient_memory_mb"]
    for r in results
]

plt.figure(figsize=(8, 5))

# Plot naive only where measurement succeeded
valid_lengths = [
    L for L, m in zip(lengths, naive)
    if m is not None
]

valid_naive = [
    m for m in naive
    if m is not None
]

plt.plot(
    valid_lengths,
    valid_naive,
    marker="o",
    label="Naive Relative Attention",
)

plt.plot(
    lengths,
    efficient,
    marker="o",
    label="Memory-Efficient Relative Attention",
)

plt.xlabel("Sequence Length")
plt.ylabel("Peak GPU Memory (MB)")

plt.title(
    "Relative Attention Memory Scaling"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plot_path = "results/memory_scaling_comparison.png"

plt.savefig(
    plot_path,
    dpi=300
)

plt.show()

print("Saved:", plot_path)