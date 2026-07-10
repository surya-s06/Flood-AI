import os

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import random


class FloodDataset(Dataset):
    """
    PyTorch Dataset for FloodAI satellite flood segmentation.

    Returns:
        image      : Tensor [3,256,256]
        mask       : Tensor [1,256,256]
        image_name : Original image filename
    """

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            file
            for file in os.listdir(image_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        self.masks = sorted([
            file
            for file in os.listdir(mask_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.images) == 0:
            raise ValueError("No images found.")

        if len(self.masks) == 0:
            raise ValueError("No masks found.")

        if len(self.images) != len(self.masks):
            raise ValueError(
                "Number of images and masks do not match."
            )

        self.image_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor()

])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (256, 256),
                interpolation=InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]
        mask_name = self.masks[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        image = Image.open(image_path).convert("RGB")

        mask = Image.open(mask_path).convert("L")
        
        # ----------------------------------------
        # Paired Horizontal Flip
        # ----------------------------------------

        if random.random() < 0.5:

            image = TF.hflip(image)

            mask = TF.hflip(mask)

        # ----------------------------------------
        # Paired Rotation
        # ----------------------------------------

        angle = random.uniform(-10, 10)

        image = TF.rotate(image, angle)

        mask = TF.rotate(
            mask,
            angle,
            interpolation=InterpolationMode.NEAREST
        )

        image = self.image_transform(image)

        mask = self.mask_transform(mask)

        return image, mask