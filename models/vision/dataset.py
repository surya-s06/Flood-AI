from torch.utils.data import Dataset
from PIL import Image
import os


class FloodDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))


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

        return image, mask