import numpy as np
import torch


def transform_tif_to_bgr_arr(tif_arr: np.ndarray) -> np.ndarray:
    """Transform tif array to bgr array"""
    arr = np.moveaxis(tif_arr, 0, -1)
    # rbg -> bgr
    return arr[:, :, [2, 1, 0]]


def unapply_batch_imagenet_normalization(batch: torch.Tensor) -> torch.Tensor:
    """Unnormalize batch of images with ImageNet with mean and std"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch.device)
    return batch * std + mean


def apply_batch_imagenet_normalization(batch: torch.Tensor) -> torch.Tensor:
    """Normalize batch of images with ImageNet mean and std"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch.device)
    return (batch - mean) / std
