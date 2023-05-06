import torch.nn as nn
from .odconv import ODConv2d
from ..builder import BACKBONES
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger

__all__ = ['OD_ResNet']


def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
        super(BasicBlock, self).__init__()
        self.conv1 = odconv3x3(inplanes, planes, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = odconv3x3(planes, planes, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
        super(Bottleneck, self).__init__()
        self.conv1 = odconv1x1(inplanes, planes, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = odconv3x3(planes, planes, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = odconv1x1(planes, planes * self.expansion, reduction=reduction, kernel_num=kernel_num)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


@BACKBONES.register_module()
class OD_ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, num_classes=1000, reduction=0.0625, kernel_num=1,
                 frozen_stages=0, out_indices=(0, 1, 2, 3), norm_eval=True, pretrained=None):
        super(OD_ResNet, self).__init__()
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=reduction, kernel_num=kernel_num)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction, kernel_num=kernel_num)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction, kernel_num=kernel_num)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction, kernel_num=kernel_num)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.norm_eval = norm_eval
        self.pretrained = pretrained

    def net_update_temperature(self, temperature):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)

    def _make_layer(self, block, planes, blocks, stride=1, reduction=0.625, kernel_num=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, reduction=reduction, kernel_num=kernel_num))
        self.inplanes = planes * block.expansion
        for idx in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=reduction, kernel_num=kernel_num))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(OD_ResNet, self).train(mode)
        self._freeze_stages()

        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=True, logger=logger, map_location='cpu')
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.net_update_temperature(1.0)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for idx in range(4):
            layer = getattr(self, f'layer{idx + 1}')
            x = layer(x)
            if idx in self.out_indices:
                outs.append(x)
        return outs

    def forward(self, x):
        return self._forward_impl(x)
