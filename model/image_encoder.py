import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import pickle
import numpy as np

import torchvision
import pdb

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = F.interpolate(x_to_upsample, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class Image_encoder_res101(nn.Module):
    def __init__(self, C, with_att=False):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

        self.with_att = with_att
        if self.with_att:
            self.positional_encoding = self._get_positional_encoding(self.C).cuda()
            self.attention = nn.MultiheadAttention(self.C, 4, dropout=0.1)
            self.fc = nn.Linear(self.C, self.C)

    def _get_positional_encoding(self, input_dim, max_len=6420):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2) * -(torch.log(torch.tensor(10000.0)) / input_dim))
        pos_encoding = torch.zeros(max_len, input_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # add batch dimension

    def forward(self, x):
        x1 = self.backbone(x) #(b, 3, 480, 854) -> (b, 512, 60, 107)
        x2 = self.layer3(x1) # (b, 512, 60, 107) ->  (b, 1024, 30, 54)
        x = self.upsampling_layer(x2, x1)  # (b, 512, 60, 107) + (b, 1024, 30, 54) -> (b, 512, 60, 107)
        x = self.depth_layer(x) #(b, C, 60, 107)
        if self.with_att:
            B, D, H, W = x.size()
            x = x.view(B, D, -1)
            x = x + self.positional_encoding[:, :H * W].repeat(B, 1, 1).permute(0, 2, 1)
            x = x.permute(2, 0, 1)
            att_output, _ = self.attention(x, x, x)
            x = self.fc(x + att_output) + x
            x = x.permute(1, 2, 0).view(B, D, H, W)
        return x

class Image_encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)
        return x

if __name__ == '__main__':
    model = Image_encoder_res101(64).cuda()
    images = torch.from_numpy(np.random.rand(4, 3, 480, 854)).float().cuda()
    out = model(images)
