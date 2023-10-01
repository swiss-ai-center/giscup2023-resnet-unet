import shutil
import time
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.multiprocessing
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.strategies import DDPStrategy
from torchsummary import summary

from pytorch.callbacks import PreviewCallback
from pytorch.dataloaders.seg_mask_data_module import SegMaskDataModule
from pytorch.dataloaders.seg_mask_dataset import SegMaskDataset
from pytorch.loss_functions import TverskyLakeDetectionLoss
from pytorch.models.resnet_unet import ResNetUNet
from utils.seed import set_seed


def train(
    *,
    seed: int,
    dataset_path: Path,
    image_size: int,
    split: float,
    batch_size: int,
    accumulate_grad_batches: int,
    precision: Optional[str],
    es_patience: Optional[int],
    lr: float,
    lr_step_size: Optional[int],
    lr_gamma: Optional[float],
    tversky_alpha: float,
    tversky_beta: float,
    lake_loss_gamma: Optional[float],
    epochs: int,
    num_workers: int,
    is_tune: bool = False,
    save_ckpt: bool = True,
    metric_monitor: str = "val_loss",
    use_best_ckpt: bool = True,
    cwd: Path = None,
) -> None:
    if is_tune:
        from ray.train.lightning import (
            RayDDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )

    set_seed(seed)

    model = ResNetUNet(
        n_classes=1,
        optimizer=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        lr_gamma=lr_gamma,
        lr_step_size=lr_step_size,
        loss_fn=TverskyLakeDetectionLoss,
        loss_fn_params=dict(
            alpha=tversky_alpha, beta=tversky_beta, gamma=lake_loss_gamma
        ),
        in_channels=3,
    )
    if not torch.cuda.is_available():
        summary(model, input_size=(3, image_size, image_size), batch_size=batch_size)
    else:
        torch.set_float32_matmul_precision("high")
    dm = SegMaskDataModule(
        dataset_cls=SegMaskDataset,
        dataset_path=dataset_path,
        image_size=image_size,
        split=split,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        cwd=cwd,
    )
    # Callbacks
    callbacks = [
        PreviewCallback(),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if save_ckpt:
        callbacks.append(
            ModelCheckpoint(
                filename="{epoch:02d}-{step}-{" + metric_monitor + ":.3f}",
                monitor=metric_monitor,
                save_top_k=1,
                mode="min",
            )
        )
    if es_patience is not None:
        callbacks.append(
            EarlyStopping(monitor=metric_monitor, patience=es_patience, mode="min"),
        )
    # Trainer
    if is_tune:
        trainer = pl.Trainer(
            max_epochs=epochs,
            strategy=RayDDPStrategy(find_unused_parameters=True),
            precision=precision if precision else "32-true",
            benchmark=True if torch.cuda.is_available() else False,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks + [RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
            limit_val_batches=0 if split == 1 else None,
        )
        trainer = prepare_trainer(trainer)
    else:
        trainer = pl.Trainer(
            max_epochs=epochs,
            precision=precision if precision else "32-true",
            strategy=DDPStrategy(
                static_graph=True,
            )
            if torch.cuda.is_available()
            else "auto",
            benchmark=True if torch.cuda.is_available() else False,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
            limit_val_batches=0 if split == 1 else None,
        )

    trainer.fit(
        model,
        datamodule=dm,
    )

    if not is_tune:
        # Copy model to root folder
        ckpt_folder = (
            sorted(
                Path("lightning_logs").glob("version_*"),
                key=lambda x: int(x.name.split("_")[-1]),
            )[-1]
            / "checkpoints"
        )
        if use_best_ckpt:
            model_path: Path = list(ckpt_folder.glob("*.ckpt"))[0]
        else:
            model_path = ckpt_folder / "last.ckpt"
            trainer.save_checkpoint(model_path)

        max_retries = 20
        retries = 0
        while True:
            if retries >= max_retries:
                raise RuntimeError("Model not found")
            if model_path.exists():
                shutil.copy(model_path, Path("out/model.ckpt"))
                break
            time.sleep(1)
            retries += 1


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"][params["prepare"]["method"]]
    train_params = params["process"]["ptl_unet"]

    train(
        seed=train_params["seed"],
        dataset_path=Path(train_params["dataset_path"]),
        image_size=prepare_params["tile_output_size"],
        split=train_params["split"],
        batch_size=train_params["batch_size"],
        accumulate_grad_batches=train_params["acc_grad_batches"],
        precision=train_params["precision"],
        metric_monitor=train_params["metric_monitor"],
        use_best_ckpt=train_params["use_best_ckpt"],
        es_patience=train_params["es_patience"],
        lr=train_params["lr"],
        lr_step_size=train_params.get("lr_step_size"),
        lr_gamma=train_params.get("lr_gamma"),
        tversky_alpha=train_params["tversky_alpha"],
        tversky_beta=train_params["tversky_beta"],
        lake_loss_gamma=train_params["lake_loss_gamma"],
        epochs=train_params["epochs"],
        num_workers=params["process_num_workers"],
    )


if __name__ == "__main__":
    main()
