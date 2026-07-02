from dataset import FloodDataset

dataset = FloodDataset(
    "dataset/images",
    "dataset/masks"
)

print(len(dataset))

image, mask = dataset[0]

print(image.size)
print(mask.size)