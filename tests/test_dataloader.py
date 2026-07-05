import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader
from models.vision.dataset import FloodDataset


dataset = FloodDataset(
    "data/satellite/images",
    "data/satellite/masks"
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

images, masks = next(iter(dataloader))

print(type(images))
print(type(masks))