import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from dataset import FloodDataset
from unet import UNet


# -----------------------
# Hyperparameters
# -----------------------

BATCH_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 5


# -----------------------
# Device
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# -----------------------
# Dataset
# -----------------------

dataset = FloodDataset(
    "../../data/satellite/images",
    "../../data/satellite/masks"
)

print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Number of batches:", len(loader))

# -----------------------
# Model
# -----------------------

model = UNet().to(device)


# -----------------------
# Loss
# -----------------------

criterion = nn.BCEWithLogitsLoss()


# -----------------------
# Optimizer
# -----------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------
# Training
# -----------------------

print("\nStarting Training...\n")

for epoch in range(EPOCHS):
    
    print("Entering epoch loop...")

    running_loss = 0.0

    for images, masks in loader:
        
       

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

       
        print(images.dtype, images.shape, images.min().item(), images.max().item())
        outputs = model(images)

       

        
        loss = criterion(outputs, masks)

        print("4. Loss =", loss.item())

        print("5. Backpropagation...")
        loss.backward()

        print("6. Optimizer step...")
        optimizer.step()

        print("7. Finished one batch")

        running_loss += loss.item()

    average_loss = running_loss / len(loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )


# -----------------------
# Save model
# -----------------------

os.makedirs("saved_models", exist_ok=True)

torch.save(
    model.state_dict(),
    "saved_models/unet_flood_model.pth"
)

print("\nTraining Complete!")

print("Model saved to:")

print("saved_models/unet_flood_model.pth")