import torch

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import FloodDataset
from unet import UNet

IMAGE_DIR = "data/satellite/Images"

MASK_DIR = "data/satellite/Masks"

MODEL_PATH = "models/vision/saved_models/V3/best_model.pth"

BATCH_SIZE = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

dataset = FloodDataset(
    IMAGE_DIR,
    MASK_DIR
)

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size

_, validation_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded.")

# ---------------------------------------
# Metrics
# ---------------------------------------

def calculate_iou(prediction, target):

    prediction = prediction.bool()
    target = target.bool()

    intersection = (prediction & target).float().sum()

    union = (prediction | target).float().sum()

    return (intersection + 1e-6) / (union + 1e-6)


def calculate_dice(prediction, target):

    prediction = prediction.bool()
    target = target.bool()

    intersection = (prediction & target).float().sum()

    return (2 * intersection + 1e-6) / (
        prediction.float().sum()
        + target.float().sum()
        + 1e-6
    )
    
total_iou = 0.0

total_dice = 0.0

total_accuracy = 0.0

num_batches = 0

with torch.no_grad():

    for images, masks in validation_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        predictions = torch.sigmoid(outputs)

        predictions = (predictions > 0.5).float()

        iou = calculate_iou(
            predictions,
            masks
        )

        dice = calculate_dice(
            predictions,
            masks
        )

        accuracy = (
            predictions == masks
        ).float().mean()

        total_iou += iou.item()

        total_dice += dice.item()

        total_accuracy += accuracy.item()

        num_batches += 1
        
print()

print("Evaluation Results")

print("------------------------")

print(
    f"IoU          : {total_iou / num_batches:.4f}"
)

print(
    f"Dice Score   : {total_dice / num_batches:.4f}"
)

print(
    f"Pixel Accuracy : {total_accuracy / num_batches:.4f}"
)