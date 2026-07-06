import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from models.vision.unet import UNet

model = UNet()

x = torch.randn(1, 3, 256, 256)

y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)