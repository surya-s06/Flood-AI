import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FloodDataset
from unet import UNet


# -------------------------
# Hyperparameters
# -------------------------

BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 10


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# -------------------------
# Dataset
# -------------------------

dataset = FloodDataset(
    "data/satellite/Images",
    "data/satellite/Masks"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Dataset size:", len(dataset))
print("Batches:", len(loader))

# -------------------------
# Model
# -------------------------

model = UNet().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Model created successfully.")

# -------------------------
# Training Loop
# -------------------------

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    running_loss = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        print("Forward pass...")
        outputs = model(images)
        print("Forward pass complete")

        print("Calculating loss...")
        loss = criterion(outputs, masks)
        print("Loss:", loss.item())

        print("Backward...")
        loss.backward()
        print("Backward complete")

        print("Optimizer step...")
        optimizer.step()
        print("Optimizer complete")

        running_loss += loss.item()

    average_loss = running_loss / len(loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )