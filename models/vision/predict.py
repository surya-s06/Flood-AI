import os

import torch
from PIL import Image
from torchvision import transforms

from unet import UNet


# =====================================================
# Configuration
# =====================================================

IMAGE_PATH = "data/satellite/Images/0.jpg"

MODEL_PATH = "models/vision/saved_models/best_model.pth"

OUTPUT_DIR = "outputs/predictions"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "prediction_0.png"
)


# =====================================================
# Device
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# =====================================================
# Create output folder
# =====================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =====================================================
# Load Model
# =====================================================

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# =====================================================
# Image Transform
# =====================================================

transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.ToTensor()

])


# =====================================================
# Load Image
# =====================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)

print("Image loaded.")


# =====================================================
# Prediction
# =====================================================

with torch.no_grad():

    output = model(input_tensor)

    prediction = torch.sigmoid(output)

    prediction = (prediction > 0.5).float()


# =====================================================
# Save Prediction
# =====================================================

prediction = prediction.squeeze()

prediction = prediction.cpu()

prediction = transforms.ToPILImage()(prediction)

prediction.save(OUTPUT_PATH)

print()

print("Prediction complete!")

print()

print("Saved to:")

print(OUTPUT_PATH)