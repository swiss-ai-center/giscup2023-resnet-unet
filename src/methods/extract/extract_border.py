import multiprocessing
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import torch
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from shapely.geometry import Polygon
from torch import nn
from tqdm import tqdm

from pytorch.models.resnet_unet import ResNetUNet
from utils.image import apply_batch_imagenet_normalization
from utils.regions import get_tif_from_region


def extract_image_region(kwargs: dict) -> tuple[list, list, list]:
    """
    Extracts the prediction polygons from a region of an image.

    Args:
        kwargs (dict): A dictionary containing the following keys:
            - image_file (str): The name of the image file to extract the region from.
            - region_num (int): The number of the region to extract.
            - tile_size (int): The size of the tiles to use for the sliding window.
            - image_size (int): Size of the images for the model.
            - batch_size (int): Batch size for prediction.
            - overlap_cnt (int): Number of overlapping pixels between tiles.
            - model_path (str): The path to the model checkpoint to use for prediction.

    Returns:
        tuple[list, list, list]: A tuple containing the following lists:
            - geometries (list): A list of shapely Polygon objects representing the contours of the extracted lakes.
            - images (list): A list of strings representing the names of the images the regions were extracted from.
            - region_nums (list): A list of integers representing the numbers of the regions that were extracted.
    """
    image_file = kwargs["image_file"]
    region_num = kwargs["region_num"]
    tile_size = kwargs["tile_size"]
    image_size = kwargs["image_size"]
    batch_size = kwargs["batch_size"]
    overlap_cnt = kwargs["overlap_cnt"]
    model_path = kwargs["model_path"]

    # Load model
    model = ResNetUNet.load_from_checkpoint(model_path, n_classes=1)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model = nn.DataParallel(model)
        model.to(device)
    model.eval()

    geometries = []
    images = []
    region_nums = []

    # Get region raster
    tif_image, transform = get_tif_from_region(
        tif_path=f"data/raw/{image_file}", region_num=region_num
    )

    # Extract image from raster
    img_arr = np.moveaxis(tif_image, 0, -1)
    img_arr = np.float32(img_arr) / 255

    # Pad with zeroes
    tmp_img = img_arr.copy()
    img_arr = np.zeros(
        (
            img_arr.shape[0] + (img_arr.shape[0] % tile_size),
            img_arr.shape[1] + (img_arr.shape[1] % tile_size),
            img_arr.shape[-1],
        )
    )
    img_arr[: tmp_img.shape[0], : tmp_img.shape[1], :] = tmp_img

    # Sliding window with 50% overlap
    overlap_size = int(tile_size * 1 / overlap_cnt)
    tiles = sliding_window_view(img_arr, (tile_size, tile_size, 3))[
        :: tile_size - overlap_size,
        :: tile_size - overlap_size,
        :,
    ]
    tiles = np.squeeze(tiles, axis=2)

    # Build region's prediction image
    pred_img = np.zeros((img_arr.shape[0], img_arr.shape[1], 1), dtype=np.float32)
    pred_img_cnts = np.zeros((img_arr.shape[0], img_arr.shape[1], 1), dtype=np.uint8)

    # 25% of tile_size border
    border_size = round(tile_size * 1 / 4)
    mask = np.zeros((tile_size - border_size * 2, tile_size - border_size * 2, 1))
    mask = np.pad(
        mask,
        (
            (border_size, border_size),
            (border_size, border_size),
            (0, 0),
        ),
        mode="constant",
        constant_values=1,
    )
    mask /= overlap_cnt * 2
    # Invert mask
    mask = np.ones((tile_size, tile_size, 1)) - mask

    batch_tiles = []
    batch_coordinates = []
    batch_tile_count = 0

    def predict_batch():
        """Helper function to predict the output for a batch of tiles."""
        batch_tensor = apply_batch_imagenet_normalization(
            torch.from_numpy(np.stack(batch_tiles)).float()
        )
        with torch.no_grad():
            preds = model(batch_tensor.to(device))
        preds = torch.sigmoid(preds)
        preds = preds.detach().cpu().numpy()

        for pred, (pred_row, pred_col) in zip(preds, batch_coordinates):
            pred = cv2.resize(np.squeeze(pred), (tile_size, tile_size))
            pred = np.expand_dims(pred, axis=2)
            y, x = pred_row * (tile_size - overlap_size), pred_col * (
                tile_size - overlap_size
            )
            pred *= mask
            pred_img_cnts[y : y + tile_size, x : x + tile_size] += 1
            pred_img[y : y + tile_size, x : x + tile_size] += pred

    for row_idx, row in enumerate(tiles):
        for col_idx, tile in enumerate(row):
            # Add to batch tiles with less than 50% nodata
            if np.sum(tile == 0) / tile.size < 0.5:
                tile = cv2.resize(tile, (image_size, image_size))
                tile = np.transpose(tile, (2, 0, 1))

                batch_tiles.append(tile)
                batch_coordinates.append((row_idx, col_idx))
                batch_tile_count += 1

            if batch_tile_count == batch_size:
                predict_batch()

                batch_tiles = []
                batch_coordinates = []
                batch_tile_count = 0

    # If there's still tiles remaining
    if batch_tile_count > 0:
        predict_batch()

    batch_tiles = []
    batch_coordinates = []
    batch_tile_count = 0

    # Average overlapping tiles by dividing by count of pred_img_cnts
    pred_img_cnts = np.where(pred_img_cnts == 0, 1, pred_img_cnts)
    pred_img = pred_img / pred_img_cnts
    # Threshold prediction
    pred_img = pred_img >= 0.5
    pred_img = pred_img.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Add contours to list
    for contour in contours:
        # Polygon needs at least 4 points
        if contour.shape[0] < 4:
            continue

        pts: np.ndarray = contour.reshape(-1, 2)
        # Apply transform to points
        pts = np.apply_along_axis(lambda x: transform * x, 1, pts)
        polygon = Polygon(pts)

        # Add polygon to data
        geometries.append(polygon)
        images.append(image_file)
        region_nums.append(region_num)

    return geometries, images, region_nums


def predict(
    *,
    images: list[str],
    model_path: Path,
    tile_size: int,
    image_size: int,
    batch_size: int,
    overlap_cnt: int,
    num_processes: int,
    out_dir_path: Path,
    crs: str,
) -> None:
    """
    Predicts lake polygons for given images using a trained model.

    Args:
        images (List[str]): List of file paths to input images.
        model_path (Path): File path to the trained model.
        tile_size (int): Size of the tiles to be extracted from the input images.
        image_size (int): Size of the images for the model.
        batch_size (int): Number of tiles to be processed in a batch.
        overlap_cnt (int): Number of pixels to overlap between adjacent tiles.
        num_processes (int): Number of processes to use for parallel processing.
        out_dir_path (Path): File path to the output directory.
        crs (str): Coordinate reference system to use for the output polygons.

    Returns:
        None: The function saves the predicted lake polygons to a geopackage file.
    """
    gdf_data = {"geometry": [], "image": [], "region_num": []}

    tasks = []
    for image_file in images:
        for region_num in range(1, 7):
            tasks.append(
                {
                    "image_file": image_file,
                    "region_num": region_num,
                    "tile_size": tile_size,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "overlap_cnt": overlap_cnt,
                    "model_path": model_path,
                }
            )

    # Run tasks in parallel
    with multiprocessing.Pool(processes=min(num_processes, len(tasks))) as pool:
        for result in tqdm(
            pool.imap_unordered(extract_image_region, tasks),
            total=len(tasks),
            desc="Processing",
        ):
            geometries, images, region_nums = result
            gdf_data["geometry"].extend(geometries)
            gdf_data["image"].extend(images)
            gdf_data["region_num"].extend(region_nums)

    # Save polygons to gpkg file
    out_dir_path.mkdir(exist_ok=True)

    # Convert to geopandas dataframe
    lake_pred_polygons = gpd.GeoDataFrame(data=gdf_data, crs=crs)
    lake_pred_polygons.to_file(out_dir_path / "lake_polygons_pred.gpkg", driver="GPKG")


def segmentation_model_prediction(**kwargs) -> None:
    """
    Predicts segmentation masks for input images using a trained segmentation model.

    Args:
        **kwargs: Keyword arguments for the prediction function, including:
            - model_path (str): Path to the trained segmentation model.
            - batch_size (int): Batch size for prediction.
            - overlap_cnt (int): Number of overlapping pixels between tiles.
            - num_processes (int): Number of processes to use for prediction.
    """
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"][params["prepare"]["method"]]
    out_dir_path = Path("out")

    predict(
        images=params["images"],
        model_path=out_dir_path / "model.ckpt",
        tile_size=prepare_params["tile_size"],
        image_size=prepare_params["tile_output_size"],
        batch_size=kwargs["batch_size"],
        overlap_cnt=kwargs["overlap_cnt"],
        num_processes=params["extract_num_workers"] or 1,
        out_dir_path=out_dir_path,
        crs=params["crs"],
    )
