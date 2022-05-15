import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import config as cfg


class Network(nn.Module):
    def __init__(self, mode='stn'):
        assert mode in ['stn', 'cnn']

        super(Network, self).__init__()
        self.mode = mode
        self.local_net = LocalNetwork()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=cfg.channel, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=cfg.height // 4 * cfg.width // 4 * 16, out_features=1024),
            nn.ReLU(),
            nn.Dropout(cfg.drop_prob),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w), (b,)
        '''
        batch_size = img.size(0)
        if self.mode == 'stn':
            transform_img = self.local_net(img)
            img = transform_img
        else:
            transform_img = None

        conv_output = self.conv(img).view(batch_size, -1)
        predict = self.fc(conv_output)
        return transform_img, predict


class LocalNetwork(nn.Module):
    def __init__(self):
        super(LocalNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=cfg.channel * cfg.height * cfg.width,
                      out_features=20),
            nn.Tanh(),
            nn.Dropout(cfg.drop_prob),
            nn.Linear(in_features=20, out_features=6),
            nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)

    def forward(self, img):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, cfg.channel, cfg.height, cfg.width)))
        img_transform = F.grid_sample(img, grid)

        return img_transform


if __name__ == '__main__':
    net = LocalNetwork()

    x = torch.randn(1, 1, 40, 40) + 1
    net(x)

