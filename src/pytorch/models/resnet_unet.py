from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from pytorch.models.unet import UNet


# Implementation modified from https://github.com/JinxiaoWang/NAU-Net/blob/main/NAU-Net.py
class ResNetUNet(UNet):
    """ResNet50 Pretrained U-Net model"""

    def __init__(self, **kwargs) -> None:
        in_channels = kwargs["in_channels"]
        super().__init__(**kwargs)

        self.base_layers = list(
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children()
        )
        # Encoder
        base_conv1 = self.base_layers[0]
        conv1 = (
            base_conv1
            if in_channels == 3
            else nn.Conv2d(
                in_channels,
                base_conv1.out_channels,
                kernel_size=base_conv1.kernel_size,
                stride=base_conv1.stride,
                padding=base_conv1.padding,
                bias=base_conv1.bias,
            )
        )
        self.encode_block0 = nn.Sequential(conv1, *self.base_layers[1:3])  # 64 out
        self.encode_block1 = nn.Sequential(*self.base_layers[3:5])  # 256 out
        self.encode_block2 = self.base_layers[5]  # 512 out
        self.encode_block3 = self.base_layers[6]  # 1024 out
        self.encode_block4 = self.base_layers[7]  # 2048 out
