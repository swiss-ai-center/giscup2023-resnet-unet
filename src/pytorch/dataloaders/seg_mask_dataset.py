from pathlib import Path
from typing import Optional

import albumentations
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class SegMaskDataset(Dataset):
    """Dataset for segmentation masks"""

    def __init__(
        self,
        *,
        img_paths: pd.DataFrame,
        transform: albumentations.Compose,
        cwd: Optional[Path] = None
    ) -> None:
        """
        Initialize the dataset

        Args:
            img_paths (pd.DataFrame): dataframe with columns "img" and "ann" containing
                the paths to the images and annotations
            transform (albumentations.Compose): albumentations transform to apply to
                both image and annotation
            cwd (Path): base path to the current working directory. Useful for
                ray tune to set the correct path
        """
        self.img_paths = img_paths
        self.transform = transform
        self.cwd = Path() if cwd is None else cwd

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        img_path: str = self.img_paths.iloc[index]["img"]
        img = cv2.imread(str(self.cwd / img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_path: str = self.img_paths.iloc[index]["ann"]
        ann = cv2.imread(str(self.cwd / ann_path), cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            transformed = self.transform(image=img, mask=ann)
            img = transformed["image"]
            ann = transformed["mask"]
            ann = ann.unsqueeze(0).float() / 255.0
        return img, ann
