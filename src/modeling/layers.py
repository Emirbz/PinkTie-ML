
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3

from src.constants import VIEWS


class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape):
        super(OutputLayer, self).__init__()
        if not isinstance(output_shape, (list, tuple)):
            output_shape = [output_shape]
        self.output_shape = output_shape
        self.flattened_output_shape = int(np.prod(output_shape))
        self.fc_layer = nn.Linear(in_features, self.flattened_output_shape)

    def forward(self, x):
        h = self.fc_layer(x)
        if len(self.output_shape) > 1:
            h = h.view(h.shape[0], *self.output_shape)
        h = F.log_softmax(h, dim=-1)
        return h


class BasicBlockV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class AllViewsGaussianNoise(nn.Module):
    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        if not self.gaussian_noise_std or not self.training:
            return x

        return {
            VIEWS.L_CC: self._add_gaussian_noise(x[VIEWS.L_CC]),
            VIEWS.L_MLO: self._add_gaussian_noise(x[VIEWS.L_MLO]),
            VIEWS.R_CC: self._add_gaussian_noise(x[VIEWS.R_CC]),
            VIEWS.R_MLO: self._add_gaussian_noise(x[VIEWS.R_MLO]),
        }

    def _add_gaussian_noise(self, single_view):
        return single_view + single_view.new(single_view.shape).normal_(std=self.gaussian_noise_std)


class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {
            view_name: self._avg_pool(view_tensor)
            for view_name, view_tensor in x.items()
        }

    @staticmethod
    def _avg_pool(single_view):
        n, c, _, _ = single_view.size()
        return single_view.view(n, c, -1).mean(-1)


class AllViewsPad(nn.Module):
    """Pad tensor across all 4 views"""

    def __init__(self):
        super(AllViewsPad, self).__init__()

    def forward(self, x, pad):
        return {
            VIEWS.L_CC: F.pad(x[VIEWS.L_CC], pad),
            VIEWS.L_MLO: F.pad(x[VIEWS.L_MLO], pad),
            VIEWS.R_CC: F.pad(x[VIEWS.R_CC], pad),
            VIEWS.R_MLO: F.pad(x[VIEWS.R_MLO], pad),
        }
