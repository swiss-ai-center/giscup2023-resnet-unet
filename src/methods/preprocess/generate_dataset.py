import random
import tarfile
from pathlib import Path

import pandas as pd
import yaml

from utils.seed import set_seed


def normalize_dataset(
    *,
    metadata: dict,
    image_sim: dict,
    images: list[str],
    training_data_regions: dict,
    lake_image_ratio: float,
) -> list[str]:
    """
    Normalize dataset by balancing the number of images with and without lakes

    Args:
        metadata (dict): Metadata of the dataset
        image_sim (dict): Image similarity hashes
        images (list[str]): List of images
        training_data_regions (dict): Training data regions
        lake_image_ratio (float): Ratio of images with lakes
    """
    image_keys = list(metadata.keys())
    picked_non_lake_image_hashes = set()
    new_image_keys = []

    for image in images:
        for region in training_data_regions[image]:
            lake_images = list(
                filter(
                    lambda i: metadata[i]["has_lake"]
                    and metadata[i]["image"] == image
                    and metadata[i]["region_num"] == region,
                    image_keys,
                )
            )
            non_lake_images = list(
                filter(
                    lambda i: not metadata[i]["has_lake"]
                    and metadata[i]["image"] == image
                    and metadata[i]["region_num"] == region,
                    image_keys,
                )
            )
            if len(lake_images) == 0 and len(non_lake_images) == 0:
                continue
            # NOTE: We assume that there are more non-lake images than lake images
            assert len(non_lake_images) > len(lake_images)

            max_non_lake_images = round(len(lake_images) / lake_image_ratio)
            non_lake_images_filtered = set()
            # Pick random non-lake images while avoiding duplicates
            while len(non_lake_images_filtered) < max_non_lake_images and len(
                non_lake_images_filtered
            ) != len(non_lake_images):
                rand_non_lake_image = random.choice(non_lake_images)
                if (
                    rand_non_lake_image not in non_lake_images_filtered
                    and image_sim[rand_non_lake_image]
                    not in picked_non_lake_image_hashes
                ):
                    non_lake_images_filtered.add(rand_non_lake_image)
                    picked_non_lake_image_hashes.add(image_sim[rand_non_lake_image])
            new_image_keys += lake_images + list(non_lake_images_filtered)

    return new_image_keys


def generate_dataset(**kwargs) -> None:
    params = yaml.safe_load(open("params.yaml", "r"))
    set_seed(kwargs["seed"])

    metadata = pd.read_csv(kwargs["metadata_path"], index_col=0).to_dict(orient="index")
    image_sim = pd.read_csv(kwargs["image_sim_path"], index_col=0).to_dict(
        orient="dict"
    )["hash"]

    images = normalize_dataset(
        metadata=metadata,
        image_sim=image_sim,
        images=params["images"],
        training_data_regions=params["training_data_regions"],
        lake_image_ratio=kwargs["lake_image_ratio"],
    )

    # Create output directory
    out_dir_path = Path(params["preprocess"]["out"])
    out_dir_path.mkdir(exist_ok=True, parents=True)

    out_img_path = Path(out_dir_path, Path(kwargs["img_path"]).name)
    out_ann_path = Path(out_dir_path, Path(kwargs["ann_path"]).name)
    out_img_path.mkdir(exist_ok=True, parents=True)
    out_ann_path.mkdir(exist_ok=True, parents=True)

    img_paths = list(map(lambda i: Path(out_img_path, i), images))
    ann_paths = list(map(lambda i: Path(out_ann_path, i), images))

    # Print summary
    print("[INFO] Summary:")
    print(f"  - Nbr. of original images: {len(metadata)}")
    print(f"  - Nbr. of filtered images: {len(images)}")

    # Save dataset to csv file
    dataset = pd.DataFrame(
        {
            "img": img_paths,
            "ann": ann_paths,
        }
    )
    dataset.to_csv(out_dir_path / "dataset.csv", index=False)

    # Untar images
    print("[INFO] Untarring images...")
    img_tar_path = kwargs["img_path"] + ".tar"
    ann_tar_path = kwargs["ann_path"] + ".tar"
    with tarfile.open(img_tar_path) as img_tar, tarfile.open(ann_tar_path) as ann_tar:
        for img_path, ann_path in zip(img_paths, ann_paths):
            img_tar.extract(img_path.name, img_path.parent)
            ann_tar.extract(ann_path.name, ann_path.parent)
