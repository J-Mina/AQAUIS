import torch
import torch.nn as nn
from torch import Tensor

#Create ResNet50 Model Class
class ResBlock50(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(ResBlock50, self).__init__()
        self.expansion = 4

        self.sub_Block1= nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.sub_Block2= nn.Sequential(
            nn.Conv2d(out_channels,out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.sub_Block3= nn.Sequential(
            nn.Conv2d(out_channels,out_channels*4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels*4)
        )
        
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        # self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.identity_downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        x = self.sub_Block1(x)
        x = self.sub_Block2(x)
        x = self.sub_Block3(x)
        x = self.relu(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x = x + identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, num_layers):
        super(ResNet, self).__init__()

        self.in_channels = 64

        if (num_layers == 18 or num_layers == 34):
            self.expansion = 1
        else:
            self.expansion = 4


        self.initBlock = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1), 
        )

        #Resnet Layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride = 1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride = 2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride = 2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(self, block, num_blocks, out_channels, stride):
        downsample = None
        layers=[]
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                                        nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1,stride=stride),
                                        nn.BatchNorm2d(out_channels*self.expansion))

        layers.append(block(self.in_channels, out_channels, downsample, stride))
        self.in_channels = out_channels*self.expansion

        for i in range(num_blocks -1):
            layers.append(block(self.in_channels, out_channels)) 
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initBlock(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def ResNet50(img_channels=3, num_classes=5):
    """
    img_channels: Number os input channels of the network. (3 -> RGB)
    num_classes: Number of classes in the data.
    """
    return ResNet(ResBlock50, [3, 4, 6, 3], img_channels, num_classes, 50)



### Implement ResNet 18/34 block
class BasicBlock(nn.Module):
    def __init__(self, in_channels: int,out_channels: int, stride = 1, expansion = 1, downsample = None):
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return  out

def ResNet18(img_channels=3, num_classes=5):
    """
    img_channels: Number os input channels of the network. (3 -> RGB)
    num_classes: Number of classes in the data.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], img_channels, num_classes, 18)


def ResNet34(img_channels=3, num_classes=5):
    """
    img_channels: Number os input channels of the network. (3 -> RGB)
    num_classes: Number of classes in the data.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], img_channels, num_classes, 34)