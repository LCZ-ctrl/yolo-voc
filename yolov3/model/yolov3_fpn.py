import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov3_backbone import Conv_BN_SiLU, ConvBlocks


class Yolov3FPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 width=1.0,
                 depth=1.0,
                 out_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        c3, c4, c5 = in_channels

        # P5 -> P4
        self.top_down_layer_1 = ConvBlocks(c5, int(512 * width))
        self.reduce_layer_1 = Conv_BN_SiLU(int(512 * width), int(256 * width), kernel_size=1)

        # P4 -> P3
        self.top_down_layer_2 = ConvBlocks(c4 + int(256 * width), int(256 * width))
        self.reduce_layer_2 = Conv_BN_SiLU(int(256 * width), int(128 * width), kernel_size=1)

        # P3
        self.top_down_layer_3 = ConvBlocks(c3 + int(128 * width), int(128 * width))

        # output proj layers
        if out_channels is not None:
            self.out_layers = nn.ModuleList([
                Conv_BN_SiLU(in_channels, out_channels, kernel_size=1)
                for in_channels in [int(128 * width), int(256 * width), int(512 * width)]
            ])
            self.out_channels = [out_channels] * 3
        else:
            self.out_layers = None
            self.out_channels = [int(128 * width), int(256 * width), int(512 * width)]

    def forward(self, features):
        # c3: [B, 256, H/8, W/8]
        # c4: [B, 512, H/16, W/16]
        # c5: [B, 1024, H/32, W/32]
        c3, c4, c5 = features

        # p5/32
        p5 = self.top_down_layer_1(c5)  # [B, 1024, H/32, W/32] -> [B, 512, H/32, W/32]

        # p4/16
        p5_up = F.interpolate(self.reduce_layer_1(p5), scale_factor=2.0)  # [B, 256, H/16, W/16]
        p4 = self.top_down_layer_2(torch.cat([c4, p5_up], dim=1))  # [B, 768, H/16, W/16] -> [B, 256, H/16, W/16]

        # p3/8
        p4_up = F.interpolate(self.reduce_layer_2(p4), scale_factor=2.0)  # [B, 128, H/8, W/8]
        p3 = self.top_down_layer_3(torch.cat([c3, p4_up], dim=1))  # [B, 384, H/8, W/8] -> [B, 128, H/8, W/8]

        # out_feats = [
        #    [B, 128, H/8, W/8],  # p3
        #    [B, 256, H/16, W/16],  # p4
        #    [B, 512, H/32, W/32]  # p5
        # ]
        out_feats = [p3, p4, p5]

        # output proj layers
        if self.out_layers is not None:
            # [[B, 256, H/8, W/8], [B, 256, H/16, W/16], [B, 256, H/32, W/32]]
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(in_channels, out_channels):
    fpn_net = Yolov3FPN(in_channels=in_channels, out_channels=out_channels, width=1.0, depth=1.0)
    return fpn_net
