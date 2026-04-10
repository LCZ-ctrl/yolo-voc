import torch
import torch.nn as nn

# ImageNet pretrained weight for DarkNet-53
model_urls = {
    "darknet53": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53_silu.pth",
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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nblocks=1):
        super().__init__()

        assert in_channels == out_channels
        self.m = nn.Sequential(*[
            Bottleneck(in_channels, out_channels, expand_ratio=0.5, shortcut=True)
            for _ in range(nblocks)
        ])

    def forward(self, x):
        return self.m(x)


class ConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        hidden_channels = out_channels // 2
        self.convs = nn.Sequential(
            Conv_BN_SiLU(in_channels, out_channels, kernel_size=1),
            Conv_BN_SiLU(out_channels, hidden_channels, kernel_size=3, padding=1),
            Conv_BN_SiLU(hidden_channels, out_channels, kernel_size=1),
            Conv_BN_SiLU(out_channels, hidden_channels, kernel_size=3, padding=1),
            Conv_BN_SiLU(hidden_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.feat_dims = [256, 512, 1024]

        # P1
        self.layer_1 = nn.Sequential(
            Conv_BN_SiLU(3, 32, kernel_size=3, padding=1),
            Conv_BN_SiLU(32, 64, kernel_size=3, padding=1, stride=2),
            ResBlock(64, 64, nblocks=1)
        )

        # P2
        self.layer_2 = nn.Sequential(
            Conv_BN_SiLU(64, 128, kernel_size=3, padding=1, stride=2),
            ResBlock(128, 128, nblocks=2)
        )

        # P3
        self.layer_3 = nn.Sequential(
            Conv_BN_SiLU(128, 256, kernel_size=3, padding=1, stride=2),
            ResBlock(256, 256, nblocks=8)
        )

        # P4
        self.layer_4 = nn.Sequential(
            Conv_BN_SiLU(256, 512, kernel_size=3, padding=1, stride=2),
            ResBlock(512, 512, nblocks=8)
        )

        # P5
        self.layer_5 = nn.Sequential(
            Conv_BN_SiLU(512, 1024, kernel_size=3, padding=1, stride=2),
            ResBlock(1024, 1024, nblocks=4)
        )

    def forward(self, x):
        c1 = self.layer_1(x)  # [B, 64, H/2, W/2]
        c2 = self.layer_2(c1)  # [B, 128, H/4, W/4]
        c3 = self.layer_3(c2)  # [B, 256, H/8, W/8]
        c4 = self.layer_4(c3)  # [B, 512, H/16, W/16]
        c5 = self.layer_5(c4)  # [B, 1024, H/32, W/32]

        outputs = [c3, c4, c5]
        return outputs


def build_backbone(model_name='darknet53', pretrained=False):
    if model_name == 'darknet53':
        backbone = DarkNet53()
        feat_dims = backbone.feat_dims

    if pretrained:
        url = model_urls[model_name]
        if url is not None:
            print('Loading pretrained weight ...')
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = backbone.state_dict()
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

            backbone.load_state_dict(checkpoint_state_dict)
        else:
            print('No backbone pretrained: DarkNet53')

    return backbone, feat_dims
