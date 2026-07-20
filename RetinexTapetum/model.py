"""RetinexTapetum model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_luminance(x):
    if x.size(1) == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def high_frequency_luminance(x, pool_size=15):
    y = rgb_to_luminance(x)
    y_low = F.avg_pool2d(y, kernel_size=pool_size, stride=1, padding=pool_size // 2)
    return y - y_low


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, 1),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            DepthwiseSeparableConv(channels, channels, activation=False),
        )

    def forward(self, x):
        return F.relu(x + self.body(x), inplace=True)


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class DecomNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.head = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.body = nn.Sequential(
            ResidualBlock(base),
            ResidualBlock(base),
            ResidualBlock(base),
            ChannelGate(base),
        )
        self.r_out = nn.Conv2d(base, 3, 3, 1, 1)
        self.l_out = nn.Conv2d(base, 3, 3, 1, 1)

    def forward(self, x):
        f = F.relu(self.head(x), inplace=True)
        f = self.body(f)
        return torch.sigmoid(self.r_out(f)), torch.sigmoid(self.l_out(f))


class TapetumAttention(nn.Module):
    def __init__(self, in_ch=7, base=32):
        super().__init__()
        self.enc = nn.Sequential(
            DepthwiseSeparableConv(in_ch, base),
            ResidualBlock(base),
            ChannelGate(base),
        )
        self.down = nn.Sequential(
            nn.AvgPool2d(2),
            DepthwiseSeparableConv(base, base * 2),
            ResidualBlock(base * 2),
        )
        self.context = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, 1, 2, dilation=2, groups=base * 2),
            nn.Conv2d(base * 2, base * 2, 1),
            nn.ReLU(inplace=True),
            ChannelGate(base * 2),
        )
        self.up = nn.Conv2d(base * 2, base, 1)
        self.fuse = nn.Sequential(
            DepthwiseSeparableConv(base * 2, base),
            nn.Conv2d(base, 3, 1),
        )

    def forward(self, x):
        e = self.enc(x)
        c = self.context(self.down(e))
        c = F.interpolate(c, size=e.shape[-2:], mode="bilinear", align_corners=False)
        c = self.up(c)
        return torch.sigmoid(self.fuse(torch.cat([e, c], dim=1)))


class LambdaMap(nn.Module):
    def __init__(self, in_ch=4, base=16, lambda_max=1.35):
        super().__init__()
        self.lambda_max = lambda_max
        self.net = nn.Sequential(
            DepthwiseSeparableConv(in_ch, base),
            ResidualBlock(base),
            nn.Conv2d(base, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, illumination, dark_prior):
        x = torch.cat([illumination, dark_prior], dim=1)
        return self.lambda_max * self.net(x) * dark_prior


class ColorRefinement(nn.Module):
    def __init__(self, in_ch=13, base=24):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConv(in_ch, base),
            ResidualBlock(base),
            ChannelGate(base),
            nn.Conv2d(base, 3, 3, 1, 1),
        )

    def forward(self, x):
        return 0.08 * torch.tanh(self.net(x))


class RetinexTapetum(nn.Module):
    def __init__(self, base=32, lambda_init=0.0, lambda_max=1.35):
        super().__init__()
        del lambda_init
        self.decom_net = DecomNet(in_ch=3, base=base)
        self.tapetum_net = TapetumAttention(in_ch=7, base=base)
        self.lambda_map_net = LambdaMap(
            in_ch=4,
            base=max(base // 2, 12),
            lambda_max=lambda_max,
        )
        self.refine_net = ColorRefinement(
            in_ch=13,
            base=max(base // 2, 16),
        )

    def forward(self, low, high=None):
        reflectance_low, illumination_low = self.decom_net(low)
        frequency_high = high_frequency_luminance(low)
        dark_prior = torch.clamp(1.0 - rgb_to_luminance(illumination_low), 0.0, 1.0)
        attention_input = torch.cat([low, illumination_low, dark_prior], dim=1)
        tapetum_attention = self.tapetum_net(attention_input)
        lambda_map = self.lambda_map_net(illumination_low, dark_prior)
        illumination_t = illumination_low * (1.0 + lambda_map * tapetum_attention)
        base_enhanced = reflectance_low * illumination_t
        refine_input = torch.cat(
            [low, base_enhanced, tapetum_attention, lambda_map, frequency_high], dim=1
        )
        residual = self.refine_net(refine_input)
        enhanced = torch.clamp(base_enhanced + residual, 0.0, 1.0)

        output = {
            "enhanced": enhanced,
            "base_enhanced": base_enhanced,
            "residual": residual,
            "reflectance_low": reflectance_low,
            "illumination_low": illumination_low,
            "tapetum_attention": tapetum_attention,
            "frequency_high": frequency_high,
            "dark_prior": dark_prior,
            "illumination_t": illumination_t,
            "lambda_map": lambda_map,
            "lambda": lambda_map.mean(),
        }

        if high is not None:
            reflectance_high, illumination_high = self.decom_net(high)
            output.update(
                {
                    "reflectance_high": reflectance_high,
                    "illumination_high": illumination_high,
                    "recon_low": torch.clamp(reflectance_low * illumination_low, 0.0, 1.0),
                    "recon_high": torch.clamp(reflectance_high * illumination_high, 0.0, 1.0),
                }
            )
        return output
