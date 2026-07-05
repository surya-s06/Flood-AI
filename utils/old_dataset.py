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

        image = Image.open(
            os.path.join(
                self.image_dir,
                self.images[index]
            )
        )

        mask = Image.open(
            os.path.join(
                self.mask_dir,
                self.masks[index]
            )
        )

        return image, mask