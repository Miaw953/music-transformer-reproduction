import os
import torch
import matplotlib.pyplot as plt

MAESTRO_CKPT = "models/model100_ckpt.pt"
POP_CKPT = "models/pop100_ckpt.pt"

os.makedirs("results", exist_ok=True)

maestro = torch.load(
    MAESTRO_CKPT,
    map_location="cpu",
    weights_only=False
)

pop = torch.load(
    POP_CKPT,
    map_location="cpu",
    weights_only=False
)

maestro_val = maestro["validation_losses"]
pop_val = pop["validation_losses"]

epochs_maestro = range(1, len(maestro_val) + 1)
epochs_pop = range(1, len(pop_val) + 1)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs_maestro,
    maestro_val,
    marker="o",
    label="MAESTRO Classical Piano"
)

plt.plot(
    epochs_pop,
    pop_val,
    marker="o",
    label="Pop1K7 Pop Piano"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss: Classical vs Pop Piano")
plt.legend()
plt.grid(True)

plt.tight_layout()

output = "results/genre_validation_loss_comparison.png"

plt.savefig(output, dpi=300)
plt.show()

print("Saved:", output)

print()
print("Final MAESTRO validation loss:", maestro_val[-1])
print("Final Pop1K7 validation loss:", pop_val[-1])