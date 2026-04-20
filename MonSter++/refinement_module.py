import torch
import torch.nn as nn


class ConfigurableRefinementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must define at least an input and output width")
        if channels[0] != 4 or channels[-1] != 1:
            raise ValueError("refinement module must start with 4 input channels and end with 1 output channel")

        layers = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if out_channels != 1:
                layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)
        final_conv = self.net[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def forward(self, disparity, left_rgb):
        residual = self.net(torch.cat([disparity, left_rgb], dim=1))
        return disparity + residual, residual


class LightweightRefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = ConfigurableRefinementModule([4, 32, 32, 1])

    def forward(self, disparity, left_rgb):
        return self.module(disparity, left_rgb)


class LargeRefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = ConfigurableRefinementModule([4, 64, 128, 64, 1])

    def forward(self, disparity, left_rgb):
        return self.module(disparity, left_rgb)
