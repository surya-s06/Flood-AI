from dataset import FloodDataset

dataset = FloodDataset(
    "dataset/images",
    "dataset/masks"
)

print(f"Number of images: {len(dataset)}")

image, mask = dataset[0]

print(image.size)
print(mask.size)