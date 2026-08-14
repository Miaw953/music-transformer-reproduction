import torch
import matplotlib.pyplot as plt
import os

checkpoint_path = "models/model100_ckpt.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

train_losses = checkpoint["train_losses"]
val_losses = checkpoint["validation_losses"]

epochs = range(1, len(train_losses) + 1)

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_losses, marker="o", label="Training Loss")
plt.plot(epochs, val_losses, marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Music Transformer Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("results/training_validation_loss.png", dpi=300)
plt.show()

print("Training losses:", train_losses)
print("Validation losses:", val_losses)
print("Saved: results/training_validation_loss.png")