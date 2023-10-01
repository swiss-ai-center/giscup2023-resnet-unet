from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from torch import nn


# Implementations from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1.0):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1, alpha=0.5, beta=0.5
    ):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class LakeDetectionLoss(nn.Module):
    """Lake Detection loss based on the number of lakes detected as blobs"""

    def __init__(self, weight=None, size_average=True):
        super(LakeDetectionLoss, self).__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        smooth=1,
        min_intersection=0.5,
        max_intersection=1.6,
    ):
        """Forward pass

        Args:
            inputs (torch.Tensor): Input tensor
            targets (torch.Tensor): Target tensor
            smooth (int, optional): Smoothing factor. Defaults to 1.
            min_intersection (float, optional): Minimum intersection between
                target and prediction. Defaults to 0.5.
            max_intersection (float, optional): Maximum intersection between
                target and prediction. Defaults to 1.6.

        Returns:
            torch.Tensor: Loss
        """
        inputs = F.sigmoid(inputs)

        # Calculate the number of TP, FP, and FN for each image in the batch
        TP = 0
        FP = 0
        FN = 0

        # NOTE: should do this on the gpu with pytorch operations
        numpy_inputs = inputs.detach().cpu().numpy()
        numpy_targets = targets.detach().cpu().numpy()

        for b in range(inputs.shape[0]):
            pred_blob_labels = numpy_inputs[b, 0, :, :]
            target_blob_labels = numpy_targets[b, 0, :, :]

            pred_labels, pred_num = measure.label(
                pred_blob_labels,
                background=0,
                return_num=True,
                connectivity=1,
            )
            target_labels, target_num = measure.label(
                target_blob_labels,
                background=0,
                return_num=True,
                connectivity=1,
            )

            taken_lakes = set()
            pairs = {}  # target -> pred
            intersections = {}  # target -> intersection
            for i in range(1, target_num + 1):
                t_area = np.sum(target_labels == i)
                for j in range(1, pred_num + 1):
                    intersection = np.sum((target_labels == i) * (pred_labels == j))
                    if (
                        min_intersection * t_area
                        < intersection
                        < max_intersection * t_area
                    ):
                        if i not in taken_lakes:
                            taken_lakes.add(i)
                            pairs[i] = j
                            intersections[i] = intersection
                        elif intersection > intersections[i]:
                            pairs[i] = j
                            intersections[i] = intersection

            # nbr of extra lakes
            FP += max(pred_num - len(pairs), 0)
            # nbr of lakes that are not taken
            FN += max(target_num - len(pairs), 0)
            # nbr of lakes that are taken
            TP += len(pairs)

        LakeDetectionLoss = (TP + smooth) / (TP + FP + FN + smooth)
        return 1 - LakeDetectionLoss


class TverskyLakeDetectionLoss(nn.Module):
    """Tversky loss weighted average with Lake Detection loss"""

    def __init__(self, weight=None, size_average=True):
        super(TverskyLakeDetectionLoss, self).__init__()
        self.TverskyLoss = None
        self.LakeDetectionLoss = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        smooth=1,
        alpha=0.5,
        beta=0.5,
        gamma=0.5,
    ) -> torch.Tensor:
        """Forward pass

        Args:
            inputs (torch.Tensor): Input tensor
            targets (torch.Tensor): Target tensor
            smooth (int, optional): Smoothing factor. Defaults to 1.
            alpha (float, optional): Tversky alpha. Defaults to 0.5.
            beta (float, optional): Tversky beta. Defaults to 0.5.
            gamma (float, optional): Weight of Tversky loss. Defaults to 0.5.

        Returns:
            torch.Tensor: Loss
        """
        if self.TverskyLoss is None:
            self.TverskyLoss = TverskyLoss()
        if self.LakeDetectionLoss is None:
            self.LakeDetectionLoss = LakeDetectionLoss()
        return gamma * self.TverskyLoss(inputs, targets, smooth, alpha, beta) + (
            1 - gamma
        ) * self.LakeDetectionLoss(inputs, targets)
