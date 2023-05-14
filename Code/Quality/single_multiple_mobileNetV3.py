import torch
import torch.nn as nn
from torch import Tensor

class ConvBlock(nn.Module):
    # Convolution Block
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 act,  # Activation function
                 groups=1,  # Depthwise conv.
                 bn=True,  # Batch normalization
                 bias=False,
                 ):
        super(ConvBlock, self).__init__()

        padding = kernel_size // 2
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.c(x)))


class SeBlock(nn.Module):
    # Squeeze and Excitation Block.
    def __init__(
            self,
            in_channels: int
    ):
        super(SeBlock, self).__init__()

        C = in_channels
        r = C // 4
        self.globpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(C, r, bias=False)
        self.fc2 = nn.Linear(r, C, bias=False)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        f = self.globpool(x)
        f = torch.flatten(f, 1)
        f = self.relu(self.fc1(f))
        f = self.hsigmoid(self.fc2(f))
        f = f[:, :, None, None]

        scale = x * f
        return scale


class BNeck(nn.Module):
    # MobileNetV3 Block
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 exp_size: int,
                 se: bool,
                 act: torch.nn.modules.activation,
                 stride: int):
        super(BNeck, self).__init__()

        self.add = in_channels == out_channels and stride == 1

        self.block = nn.Sequential(
            ConvBlock(in_channels, exp_size, 1, 1, act),
            ConvBlock(exp_size, exp_size, kernel_size, stride, act, exp_size),
            SeBlock(exp_size) if se == True else nn.Identity(),
            ConvBlock(exp_size, out_channels, 1, 1, act=nn.Identity())
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        if self.add:
            res = res + x

        return res


class MobileNetV3(nn.Module):
    def __init__(
            self,
            config_name: str,
            in_channels=3,
            binary_classes=2,  # Number of classes for binary heads
            multiclass_classes=3  # Number of classes for the multiclass head
    ):
        super(MobileNetV3, self).__init__()

        config = self.config(config_name)

        # First convolutional layer
        self.conv = ConvBlock(in_channels, 16, kernel_size=3, stride=2, act=nn.Hardswish())

        # Bneck blocks in a list
        self.blocks = nn.ModuleList([])
        for c in config:
            kernel_size, exp_size, in_channels, out_channels, se, nl, s = c
            self.blocks.append(BNeck(in_channels, out_channels, kernel_size, exp_size, se, nl, s))

        #Classifier
        last_outchannel = config[-1][3]
        last_exp = config[-1][1]
        out = 1280 if config_name == "large" else 1024

        # Binary heads
        self.binary_head1 = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, binary_classes),
            nn.Sigmoid()
        )

        self.binary_head2 = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, binary_classes),
            nn.Sigmoid()
        )

        # Multiclass head
        self.multiclass_head = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, multiclass_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)

        binary_head1_output = self.binary_head1(x)
        binary_head2_output = self.binary_head2(x)
        multiclass_head_output = self.multiclass_head(x)

        return binary_head1_output, binary_head2_output, multiclass_head_output
            

    def config(self, name):
        HE, RE = nn.Hardswish(), nn.ReLU()
        # [kernel, exp_size, in_channels, out_channels, SEBlock(SE), activation_function(NL), stride(s)]
        large = [
            [3, 16, 16, 16, False, RE, 1],
            [3, 64, 16, 24, False, RE, 2],
            [3, 72, 24, 24, False, RE, 1],
            [5, 72, 24, 40, True, RE, 2],
            [5, 120, 40, 40, True, RE, 1],
            [5, 120, 40, 40, True, RE, 1],
            [3, 240, 40, 80, False, HE, 2],
            [3, 200, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 480, 80, 112, True, HE, 1],
            [3, 672, 112, 112, True, HE, 1],
            [5, 672, 112, 160, True, HE, 2],
            [5, 960, 160, 160, True, HE, 1],
            [5, 960, 160, 160, True, HE, 1]
        ]

        small = [
            [3, 16, 16, 16, True, RE, 2],
            [3, 72, 16, 24, False, RE, 2],
            [3, 88, 24, 24, False, RE, 1],
            [5, 96, 24, 40, True, HE, 2],
            [5, 240, 40, 40, True, HE, 1],
            [5, 240, 40, 40, True, HE, 1],
            [5, 120, 40, 48, True, HE, 1],
            [5, 144, 48, 48, True, HE, 1],
            [5, 288, 48, 96, True, HE, 2],
            [5, 576, 96, 96, True, HE, 1],
            [5, 576, 96, 96, True, HE, 1]
        ]

        if name == 'large':
            return large
        if name == 'small':
            return small


def MobileNetSmall(binary_classes=2, multiclass_classes=3):
    return MobileNetV3("small", binary_classes=binary_classes, multiclass_classes=multiclass_classes)

def MobileNetLarge(binary_classes=2, multiclass_classes=3):
    return MobileNetV3("large", binary_classes=binary_classes, multiclass_classes=multiclass_classes)
