import math

import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt

from pytorch.dataloaders.seg_mask_data_module import SegMaskDataModule
from pytorch.dataloaders.seg_mask_dataset import SegMaskDataset
from utils.seed import set_seed

dataset_path = "data/preprocessed/dataset.csv"
image_size = 256
split = 0.8
batch_size = 16
seed = 412
num_workers = 0

set_seed(seed)

dm = SegMaskDataModule(
    dataset_cls=SegMaskDataset,
    dataset_path=dataset_path,
    image_size=image_size,
    split=split,
    batch_size=batch_size,
    seed=seed,
    num_workers=num_workers,
)
dm.train_transform = A.Compose(
    [t for t in dm.train_transform if not isinstance(t, (A.Normalize))]
)
dm.setup()

# create plot with sample images
for batch in dm.train_dataloader():
    x, y = batch
    # plot all images on batch // 2 x batch // 2 grid
    disp = round(math.sqrt(batch_size))
    fig, axs = plt.subplots(disp, disp, figsize=(20, 20))
    fig.tight_layout()
    for i in range(batch_size):
        image = x[i].numpy().transpose(1, 2, 0).copy()
        print(x[i].shape, x[i].dtype)
        print(y[i].shape, y[i].dtype)
        mask = y[i].numpy().squeeze()
        has_lake = np.any(mask) > 0
        if has_lake:
            cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, cnts, -1, (255, 0, 0), 1)
        axs[i // disp, i % disp].imshow(image)
        axs[i // disp, i % disp].axis("off")
        axs[i // disp, i % disp].set_facecolor("black")
        if has_lake:
            axs[i // disp, i % disp].set_title("LAKE")
    # save plot to file
    plt.savefig("sample_images.png")
    input("Press Enter to continue...")
