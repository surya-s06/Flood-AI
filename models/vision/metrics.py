import torch


def pixel_accuracy(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (prediction > 0.5).float()

    correct = (prediction == target).float().sum()

    total = target.numel()

    return correct / total