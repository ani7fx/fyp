import torch
import torch.nn as nn


class LightweightRefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        final_conv = self.net[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def forward(self, disparity, left_rgb):
        residual = self.net(torch.cat([disparity, left_rgb], dim=1))
        return disparity + residual, residual
