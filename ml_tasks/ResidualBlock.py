import torch
import torch.nn as nn
import numpy as np
import os
import random

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Инициализируйте необходимое количество conv слоёв
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=3, padding=1, stride=1)

        # Инициализируйте необходимое количество batchnorm слоёв
        self.batch_n = nn.BatchNorm2d(in_channels)

        # Инициализируйте relu слой
        self.relu = nn.ReLU()

    def forward(self, x):
        mem_x = x
        x = self.conv1(x)
        x = self.batch_n(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_n(x)
        out = x
        out += mem_x
        out = self.relu(out)
        return out



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything()
input_image = torch.randn(1, 1, 3, 3)

residualblock = ResidualBlock(1)
result = residualblock(input_image)

print(torch.allclose(result, torch.tensor([[[[0.1642, 0.0969, 0.0000],
          [0.9133, 0.0000, 0.0000],
          [3.6363, 0.0000, 0.9017]]]]), atol=1e-4
              ))