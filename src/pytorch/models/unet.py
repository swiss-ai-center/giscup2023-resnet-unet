from typing import Any, Optional

import pytorch_lightning as pl
import torch
from torch import nn

from pytorch.loss_functions import DiceLoss, IoULoss
from pytorch.models.unet_layers import Conv2dReLU, DecoderBlock, EncoderBlock


class UNet(pl.LightningModule):
    """U-Net base model"""

    def __init__(
        self,
        *,
        n_classes: int,
        # Hyperparameters
        loss_fn: nn.Module,
        loss_fn_params: Optional[dict[str, Any]] = None,
        optimizer: torch.optim.Optimizer,
        optimizer_params: Optional[dict[str, Any]] = None,
        lr: Optional[float] = None,
        lr_step_size: Optional[int] = None,
        lr_gamma: Optional[float] = None,
        in_channels: int = 3,
    ) -> None:
        """Init U-Net model

        Args:
            n_classes (int): Number of output classes
            loss_fn (nn.Module): Loss function to use
            optimizer (torch.optim.Optimizer): Optimizer to use
            loss_fn_params (Optional[dict[str, Any]], optional): Optional arguments for loss function. Defaults to None.
            optimizer_params (Optional[dict[str, Any]], optional): Optional arguments for optimizer. Defaults to None.
            lr (Optional[float], optional): Custom learning rate override. Usefull for lr_find. Defaults to None.
            lr_step_size (Optional[int], optional): Learning rate StepLR step size. Defaults to None.
            lr_gamma (Optional[float], optional): Learning rate StepLR gamma. Defaults to None.
            in_channels (int, optional): Number of input channels. Defaults to 3.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["n_classes"])

        self.loss_fn = self.hparams.loss_fn()
        self.loss_fn_params = self.hparams.loss_fn_params or {}

        # loss functions to log
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()

        self.optimizer = self.hparams.optimizer
        self.optimizer_params = self.hparams.optimizer_params or {}
        self.lr = lr
        if self.lr:
            self.optimizer_params["lr"] = self.lr
        self.lr_gamma = self.hparams.lr_gamma
        self.lr_step_size = self.hparams.lr_step_size

        self.encode_block0 = EncoderBlock(in_channels, 64)
        self.encode_block1 = EncoderBlock(64, 256)
        self.encode_block2 = EncoderBlock(256, 512)
        self.encode_block3 = EncoderBlock(512, 1024)
        self.encode_block4 = EncoderBlock(1024, 2048)
        # Decoder
        self.decode_block4 = DecoderBlock(2048, 1024, 256)
        self.decode_block3 = DecoderBlock(256, 512, 128)
        self.decode_block2 = DecoderBlock(128, 256, 64)
        self.decode_block1 = DecoderBlock(64, 64, 64)
        self.decode_block0 = DecoderBlock(64, 32, 32)
        # Shallow input
        self.shallow = Conv2dReLU(in_channels, 32, 3, 1)
        # Mask prediction
        self.conv_last2 = nn.Conv2d(32, n_classes, 3, 1, 1)

    def forward(self, input) -> torch.Tensor:
        layer_shallow = self.shallow(input)

        encode_block0 = self.encode_block0(input)
        encode_block1 = self.encode_block1(encode_block0)
        encode_block2 = self.encode_block2(encode_block1)
        encode_block3 = self.encode_block3(encode_block2)
        encode_block4 = self.encode_block4(encode_block3)

        # Skip connections
        x = self.decode_block4(encode_block4, encode_block3)
        x = self.decode_block3(x, encode_block2)
        x = self.decode_block2(x, encode_block1)
        x = self.decode_block1(x, encode_block0)
        x = self.decode_block0(x, layer_shallow)

        out1 = self.conv_last2(x)
        return out1

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        if self.lr_gamma:
            return [optimizer], [
                torch.optim.lr_scheduler.StepLR(
                    optimizer=optimizer,
                    step_size=self.lr_step_size,
                    gamma=self.lr_gamma,
                )
            ]
        return optimizer

    def _compute_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor, sync_dist=False
    ) -> torch.Tensor:
        return self.loss_fn(y_hat, y, **self.loss_fn_params)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        loss = self._compute_loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice_loss", self.dice_loss(y_hat, y))
        self.log("train_iou_loss", self.iou_loss(y_hat, y))

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        loss = self._compute_loss(y_hat, y, sync_dist=True)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_dice_loss", self.dice_loss(y_hat, y), sync_dist=True)
        self.log("val_iou_loss", self.iou_loss(y_hat, y), sync_dist=True)
        # Tensorboard hyperparameter metric
        self.log("hp_metric", loss, sync_dist=True)

        return loss
