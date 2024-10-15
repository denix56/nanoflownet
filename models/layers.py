import torch
from torch import nn
import torch.nn.functional as F


class Conv2dBeforeSigmoid(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dBeforeSigmoid, self).__init__(*args, **kwargs)


def act_func(name: str):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError()


class Conv2dSame(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        # Inherit from nn.Conv2d but set padding to 0 (we will handle it manually)
        super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, input):
        # Get input dimensions and kernel dimensions
        batch_size, in_channels, in_height, in_width = input.size()
        out_channels, _, kernel_height, kernel_width = self.weight.size()

        # Calculate the effective kernel size considering dilation
        effective_kernel_height = (kernel_height - 1) * self.dilation[0] + 1
        effective_kernel_width = (kernel_width - 1) * self.dilation[1] + 1

        out_height = (in_height + self.stride[0] - 1) // self.stride[0]
        out_width = (in_width + self.stride[1] - 1) // self.stride[1]

        # Calculate the padding for 'same' behavior
        padding_height = max((out_height - 1) * self.stride[0] + effective_kernel_height - in_height, 0)
        padding_width = max((out_width - 1) * self.stride[1] + effective_kernel_width - in_width, 0)

        # Split the padding into top/bottom and left/right
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        # Pad the input manually using F.pad
        input_padded = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))

        # Apply convolution using the original forward function of nn.Conv2d
        return F.conv2d(input_padded, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        # Depthwise convolution (in_channels separate convolutions)
        self.depthwise = Conv2dSame(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels,
                                   bias=bias)

        # Pointwise convolution (1x1 convolution to combine depthwise outputs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, dilation=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AvgPool2dSamePadding(nn.Module):
    def __init__(self, kernel_size, stride):
        super(AvgPool2dSamePadding, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Calculate padding
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size

        if isinstance(self.stride, int):
            stride = (self.stride, self.stride)
        else:
            stride = self.stride

        batch_size, in_channels, in_height, in_width = x.size()

        out_height = (in_height + stride[0] - 1) // stride[0]
        out_width = (in_width + stride[1] - 1) // stride[1]

        # Calculate the padding for 'same' behavior
        padding_height = max((out_height - 1) * stride[0] + kernel_size[0] - in_height, 0)
        padding_width = max((out_width - 1) * stride[1] + kernel_size[1] - in_width, 0)

        # Split the padding into top/bottom and left/right
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        # Pad the input manually using F.pad
        input_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom))

        # Apply average pooling
        output = F.avg_pool2d(input_padded, kernel_size=kernel_size, stride=stride)

        return output


class STDCBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int, act: str):
        super(STDCBlock, self).__init__()

        layers = [nn.Sequential(nn.Conv2d(in_channels, int(filters / 2), 1, padding='same', bias=True),
                                act_func(act)),
                  nn.Sequential(SeparableConv2d(int(filters / 2), int(filters / 4), 3, padding='same', stride=1, bias=True),
                                act_func(act)),
                  nn.Sequential(SeparableConv2d(int(filters / 4), int(filters / 8), 3, padding='same', stride=1, bias=True),
                                act_func(act)),
                  nn.Sequential(SeparableConv2d(int(filters / 8), int(filters / 8), 3, padding='same', stride=1, bias=True),
                                act_func(act))]
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        fusion = []
        for layer in self.net:
            x = layer(x)
            fusion.append(x)
        return torch.cat(fusion, dim=1)


class STDCBlockV2(nn.Module):
    def __init__(self, in_channels: int, filters: int, act: str):
        super(STDCBlockV2, self).__init__()

        self.conv1 = nn.Sequential(
            SeparableConv2d(in_channels, int(filters / 2), 3, padding='same', stride=2, bias=True),
            act_func(act))
        self.pooled_conv1 = nn.Sequential(
                      AvgPool2dSamePadding(3, stride=2),
                      nn.Conv2d(in_channels, int(filters / 2), 1, padding='same', bias=True),
                      act_func(act))
        self.conv2 = nn.Sequential(
            SeparableConv2d(int(filters / 2), int(filters / 4), 3, padding='same', stride=1, bias=True),
            act_func(act))
        self.conv3 = nn.Sequential(
            SeparableConv2d(int(filters / 4), int(filters / 4), 3, padding='same', stride=1, bias=True),
            act_func(act))

    def forward(self, x):
        conv1 = self.conv1(x)
        pooled_conv1 = self.pooled_conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        return torch.cat([pooled_conv1, conv2, conv3], dim=1)


class StageBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int, act: str):
        super(StageBlock, self).__init__()
        self.block = nn.Sequential(STDCBlockV2(in_channels, filters, act),
                                   STDCBlock(filters, filters, act))

    def forward(self, x):
        return self.block(x)


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels: int, filters: int, act: str):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Sequential(
            SeparableConv2d(in_channels, filters, kernel_size=3, padding='same', bias=True),
            act_func(act))

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv2dBeforeSigmoid(filters, filters, kernel_size=1, stride=1, padding='same', bias=True),
            act_func('sigmoid')
        )

    def forward(self, x):
        x = self.conv(x)
        x = x * self.attn(x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels: int, filters: int, act: str):
        super(FeatureFusionModule, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, filters, 1, padding='same', bias=True),
                      act_func(act))
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(filters, filters // 4, kernel_size=1, stride=1, padding='same', bias=True),
            act_func(act),
            Conv2dBeforeSigmoid(filters // 4, filters, kernel_size=1, stride=1, padding='same', bias=True),
            act_func('sigmoid')
        )

    def forward(self, context, spatial):
        features = torch.cat([context, spatial], dim=1)
        features = self.conv(features)
        features_attention = features * self.attn(features)
        return features_attention + features


class FlowHead(nn.Module):
    def __init__(self, in_channels: int, filters: int, output_filters: int, act: str, before_sigmoid: bool = False):
        super(FlowHead, self).__init__()
        self.net = nn.Sequential(
            SeparableConv2d(in_channels, filters, kernel_size=3, padding='same', bias=True),
            act_func(act),
            (Conv2dBeforeSigmoid if before_sigmoid else nn.Conv2d)(filters, output_filters, kernel_size=1, padding='same', bias=True)
        )

    def forward(self, x):
        return self.net(x)
