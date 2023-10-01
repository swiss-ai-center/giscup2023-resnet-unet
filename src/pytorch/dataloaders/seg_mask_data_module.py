import random
from pathlib import Path
from typing import Optional

import albumentations as A
import albumentations.pytorch
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pytorch.dataloaders.seg_mask_dataset import SegMaskDataset


def seed_worker(worker_id):
    """
    Helper function to seed workers with different seeds for
    reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SegMaskDataModule(pl.LightningDataModule):
    """Lightning Data module for segmentation masks"""

    def __init__(
        self,
        *,
        dataset_cls: SegMaskDataset,
        dataset_path: Path,
        image_size: int,
        split: float,
        batch_size: int,
        seed: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        cwd: Optional[Path] = None,
    ) -> None:
        """
        Initialize the data module

        Args:
            dataset_cls (SegMaskDataset): dataset class to use
            dataset_path (Path): path to the dataset csv file
            image_size (int): size of the images
            split (float): train/val split
            batch_size (int): batch size
            seed (int): seed for reproducibility
            num_workers (int): number of workers to use for data loading. They will
                be split evenly between train and val
            cwd (Path): base path to the current working directory. Useful for
                ray tune to set the correct path
        """
        super().__init__()
        self.dataset_cls = dataset_cls
        self.dataset_path = dataset_path
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cwd = Path() if cwd is None else cwd
        self.gen = torch.Generator().manual_seed(self.seed)
        # Transforms
        self.train_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.OneOf(
                    [
                        A.HorizontalFlip(),
                        A.VerticalFlip(),
                        A.Compose(
                            [
                                A.HorizontalFlip(always_apply=True),
                                A.VerticalFlip(always_apply=True),
                            ]
                        ),
                    ],
                    p=0.25,
                ),
                A.OneOf(
                    [
                        A.Rotate(180),
                        A.Perspective(
                            scale=(0.05, 0.1),
                            pad_mode=cv2.BORDER_CONSTANT,
                            mask_pad_val=0,
                        ),
                        A.Compose(
                            [
                                A.Perspective(
                                    scale=(0.05, 0.1),
                                    pad_mode=cv2.BORDER_CONSTANT,
                                    mask_pad_val=0,
                                    always_apply=True,
                                ),
                                A.Rotate(
                                    180,
                                    always_apply=True,
                                ),
                            ]
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Sharpen(alpha=(0.1, 0.3)),
                        A.Blur(blur_limit=3),
                    ],
                    p=0.15,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.25, contrast_limit=0.25
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=5,
                            sat_shift_limit=10,
                            val_shift_limit=10,
                        ),
                    ],
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        )
        self.val_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        )

    def setup(self, stage: str = None) -> None:
        img_paths = pd.read_csv(self.cwd / self.dataset_path)
        # Shuffle the dataset
        img_paths = img_paths.sample(frac=1, random_state=self.seed)
        # Perform train/val split
        train_img_paths = img_paths[: int(self.split * len(img_paths))]
        val_img_paths = img_paths[int(self.split * len(img_paths)) :]

        self.train_split = self.dataset_cls(
            img_paths=train_img_paths,
            transform=self.train_transform,
            cwd=self.cwd,
        )
        self.val_split = self.dataset_cls(
            img_paths=val_img_paths,
            transform=self.val_transform,
            cwd=self.cwd,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=self.gen,
            num_workers=self.num_workers // 2,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=self.gen,
            num_workers=self.num_workers // 2,
            pin_memory=self.pin_memory,
        )
