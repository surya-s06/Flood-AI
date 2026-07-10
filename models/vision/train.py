import os
import csv
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import FloodDataset
from unet import UNet
from loss import BCEDiceLoss

# ======================================================
# Configuration
# ======================================================

IMAGE_DIR = "data/satellite/Images"
MASK_DIR = "data/satellite/Masks"

BATCH_SIZE = 3
LEARNING_RATE = 0.001
EPOCHS = 10

TRAIN_SPLIT = 0.8

SAVE_DIR = "models/vision/saved_models/V2"

HISTORY_FILE = "models/vision/saved_models/V2/history.csv"


# ======================================================
# Device
# ======================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# ======================================================
# Create folders
# ======================================================

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("outputs/training", exist_ok=True)


# ======================================================
# Dataset
# ======================================================

full_dataset = FloodDataset(
    IMAGE_DIR,
    MASK_DIR
)

train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Total Images      : {len(full_dataset)}")
print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")


# ======================================================
# Model
# ======================================================

model = UNet().to(device)

criterion = BCEDiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

print("\nModel initialized successfully.")

print("Checkpoint A")


# ======================================================
# CSV History
# ======================================================

with open(HISTORY_FILE, "w", newline="") as file:
    
    print("Checkpoint B")

    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Train Loss",
        "Validation Loss"
    ])


# ======================================================
# Best validation loss
# ======================================================

best_val_loss = float("inf")

print("Checkpoint C")

# ======================================================
# Validation Function
# ======================================================

def validate():

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            running_val_loss += loss.item()

    average_val_loss = running_val_loss / len(val_loader)

    return average_val_loss

# ======================================================
# Training
# ======================================================

print("\n========================================")
print("Training Started")
print("========================================\n")

print("Checkpoint D")

for epoch in range(EPOCHS):
    
    print(f"Starting Epoch {epoch+1}")

    model.train()

    running_train_loss = 0.0

    for images, masks in train_loader:
        
        print("Loaded first batch")

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        print("1. Running forward pass...")

        print(images.dtype, images.shape, images.min().item(), images.max().item())

        outputs = model(images)

        print("2. Forward pass complete")

        print("3. Calculating loss...")

        loss = criterion(outputs, masks)

        print("4. Loss =", loss.item())

        print("5. Backpropagation...")

        loss.backward()

        print("6. Backpropagation complete")

        torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
        )

        print("7. Optimizer step...")

        optimizer.step()

        print("8. Optimizer complete")

        running_train_loss += loss.item()

    average_train_loss = running_train_loss / len(train_loader)

    average_val_loss = validate()
    
    scheduler.step(average_val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Validation Loss: {average_val_loss:.4f}"
    )
    
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Learning Rate : {current_lr:.6f}")

    # ==========================================
    # Save history
    # ==========================================

    with open(HISTORY_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            epoch + 1,
            average_train_loss,
            average_val_loss
        ])

    # ==========================================
    # Save checkpoint every epoch
    # ==========================================

    epoch_path = os.path.join(
        SAVE_DIR,
        f"epoch_{epoch+1}.pth"
    )

    torch.save(
        model.state_dict(),
        epoch_path
    )

    # ==========================================
    # Save best model
    # ==========================================

    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        best_path = os.path.join(
            SAVE_DIR,
            "best_model.pth"
        )

        torch.save(
            model.state_dict(),
            best_path
        )

        print("✓ Best model updated")

print("\n========================================")
print("Training Finished")
print("========================================")

# ======================================================
# Save last model
# ======================================================

last_model_path = os.path.join(
    SAVE_DIR,
    "last_model.pth"
)

torch.save(
    model.state_dict(),
    last_model_path
)

print("\nSaved:")
print("• last_model.pth")
print("• best_model.pth")
print("• epoch_1 ... epoch_10")
print("\nHistory saved to:")
print(HISTORY_FILE)
print("\nDone!")