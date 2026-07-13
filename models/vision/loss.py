import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)

        targets = targets.view(-1)

        intersection = (
            predictions * targets
        ).sum()

        dice = (
            2 * intersection + 1
        ) / (
            predictions.sum()
            + targets.sum()
            + 1
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, predictions, targets):

        bce_loss = self.bce(
            predictions,
            targets
        )

        dice_loss = self.dice(
            predictions,
            targets
        )

        return bce_loss + dice_loss