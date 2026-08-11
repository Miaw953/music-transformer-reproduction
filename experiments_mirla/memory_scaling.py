import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import MusicTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

sequence_lengths = [128, 256, 512, 1024]

model = MusicTransformer().to(device)
model.eval()

print("Model loaded.")
print("Sequence lengths:", sequence_lengths)