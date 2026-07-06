import sys
import os

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.vision.dataset import FloodDataset

dataset = FloodDataset(
    "data/satellite/images",
    "data/satellite/masks"
)

image, mask = dataset[0]

print(type(image))
print(type(mask))

print(image.shape)
print(mask.shape)