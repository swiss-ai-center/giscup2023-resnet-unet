import pytorch_lightning as pl

from utils.image import unapply_batch_imagenet_normalization


class PreviewCallback(pl.Callback):
    """Lightning callback to save a preview of the model's predictions"""

    def __init__(self) -> None:
        super().__init__()
        self.batch: tuple = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Called when the validation epoch ends"""

        if self.batch is None:
            val_dataloader = trainer.datamodule.val_dataloader()
            # Grab a few validation samples from the dataloader
            self.batch = next(iter(val_dataloader))

        x, y = self.batch
        # Forward pass
        y_hat = pl_module(x.to(pl_module.device))

        max_images = 32
        if trainer.global_step == 0:  # only save input and target images the first time
            trainer.logger.experiment.add_images(
                "input",
                unapply_batch_imagenet_normalization(x[:max_images, :3, :, :]),
                trainer.global_step,
            )
            trainer.logger.experiment.add_images(
                "target", y[:max_images], trainer.global_step
            )
        trainer.logger.experiment.add_images(
            "pred", y_hat[:max_images], trainer.global_step
        )
