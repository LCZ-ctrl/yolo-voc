import torch
import torch.nn as nn
from .yolov5_neck import SPPF

model_urls = {
    "cspdarknet_nano": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_nano.pth",
    "cspdarknet_small": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_small.pth",
    "cspdarknet_medium": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_medium.pth",
    "cspdarknet_large": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_large.pth",
    "cspdarknet_huge": None,
}


class Conv_BN_SiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, shortcut=False):
        super().__init__()

        hidden_channels = int(out_channels * expand_ratio)
        self.cv1 = Conv_BN_SiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = Conv_BN_SiLU(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        h = self.cv2(self.cv1(x))
        return x + h if self.shortcut else h


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, nblocks=1, shortcut=False):
        super().__init__()

        hidden_channels = int(out_channels * expand_ratio)
        self.cv1 = Conv_BN_SiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = Conv_BN_SiLU(in_channels, hidden_channels, kernel_size=1)
        self.m = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, expand_ratio=1.0, shortcut=shortcut)
            for _ in range(nblocks)
        ])
        self.cv3 = Conv_BN_SiLU(2 * hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x1)
        out = self.cv3(torch.cat([x3, x2], dim=1))

        return out


class CSPDarkNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0):
        super().__init__()
        self.feat_dims = [int(256 * width), int(512 * width), int(1024 * width)]

        # P1
        self.layer_1 = Conv_BN_SiLU(3, int(64 * width), kernel_size=6, padding=2, stride=2)

        # P2
        self.layer_2 = nn.Sequential(
            Conv_BN_SiLU(int(64 * width), int(128 * width), kernel_size=3, padding=1, stride=2),
            CSPBlock(int(128 * width), int(128 * width), expand_ratio=0.5, nblocks=int(3 * depth), shortcut=True)
        )

        # P3
        self.layer_3 = nn.Sequential(
            Conv_BN_SiLU(int(128 * width), int(256 * width), kernel_size=3, padding=1, stride=2),
            CSPBlock(int(256 * width), int(256 * width), expand_ratio=0.5, nblocks=int(9 * depth), shortcut=True)
        )

        # P4
        self.layer_4 = nn.Sequential(
            Conv_BN_SiLU(int(256 * width), int(512 * width), kernel_size=3, padding=1, stride=2),
            CSPBlock(int(512 * width), int(512 * width), expand_ratio=0.5, nblocks=int(9 * depth), shortcut=True)
        )

        # P5
        self.layer_5 = nn.Sequential(
            Conv_BN_SiLU(int(512 * width), int(1024 * width), kernel_size=3, padding=1, stride=2),
            SPPF(int(1024 * width), int(1024 * width), expand_ratio=0.5),
            CSPBlock(int(1024 * width), int(1024 * width), expand_ratio=0.5, nblocks=int(3 * depth), shortcut=True)
        )

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]
        return outputs


def load_weight(model, model_name):
    # load weight
    print('Loading pretrained weight ...')
    url = model_urls[model_name]
    if url is not None:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                # print(k)

        model.load_state_dict(checkpoint_state_dict)
    else:
        print('No pretrained for {}'.format(model_name))

    return model


def build_backbone(model_name, pretrained=False):
    if model_name == 'cspdarknet_nano':
        backbone = CSPDarkNet(width=0.25, depth=0.34)
        if pretrained:
            backbone = load_weight(backbone, model_name)
    elif model_name == 'cspdarknet_small':
        backbone = CSPDarkNet(width=0.5, depth=0.34)
        if pretrained:
            backbone = load_weight(backbone, model_name)
    elif model_name == 'cspdarknet_medium':
        backbone = CSPDarkNet(width=0.75, depth=0.67)
        if pretrained:
            backbone = load_weight(backbone, model_name)
    elif model_name == 'cspdarknet_large':
        backbone = CSPDarkNet(width=1.0, depth=1.0)
        if pretrained:
            backbone = load_weight(backbone, model_name)
    elif model_name == 'cspdarknet_huge':
        backbone = CSPDarkNet(width=1.25, depth=1.34)
        if pretrained:
            backbone = load_weight(backbone, model_name)

    feat_dims = backbone.feat_dims
    return backbone, feat_dims
