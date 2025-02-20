import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import random

import copy
import math
import numpy as np
from typing import List, Optional
from model.plugin import MLP
from model.netvlad import NetVLADLoupe
from model.transformer.place_transformer import TransformerPlaceDecoder
from model.transformer.rpe_radius_transformer import RPERadiusTransformerLayer
from model.transformer.positional_embedding import RelativeAngleEmbedding, RelativeDistanceEmbedding, RelativePositionEmbedding
from torchvision.transforms.functional import rotate
import pdb
import time

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=True):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fftshift2d(x):
    for dim in range(1, len(x.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift = n_shift + 1  # for odd-sized images
        x = roll_n(x, axis=dim, n=n_shift)
    return x  # last dim=2 (real&imag)

class Bev_desc_decoder(nn.Module):
    def __init__(self, cfg, with_normalize=True, training=False, rotate_aug=True):
        super().__init__()
        self.traning = training
        self.rotate_aug = rotate_aug

        self.input_size = (cfg.model.feature_dim, cfg.model.Z, cfg.model.X)
        self.output_size = cfg.model.desc_dim
        self.with_local_feats = cfg.model.with_local_feats
        self.with_normalize = with_normalize
        self.pose_estimate = cfg.localizer.pose_estimate
        self.loop_detect = cfg.localizer.loop_detect

        # pooling local descs to global descs with GeM
        self.theta = cfg.model.theta
        self.radius = cfg.model.radius
        if self.loop_detect == 'distance' or self.loop_detect == 'pdistance' :
            self.layer2 = torchvision.models.resnet101(pretrained=False).layer2
            self.pooling = GeM()
        elif self.loop_detect == 'disco':
            self.disco_conv = nn.Conv2d(in_channels=self.input_size[0], out_channels=1, kernel_size=3, padding=1, stride=1)
        elif self.loop_detect == 'netvlad':
            self.netvlad = NetVLADLoupe(feature_size=256, max_samples=int(self.theta * self.radius), cluster_size=64, output_dim=256, gating=True, add_batch_norm=False)
            self.fc_out = nn.Linear(256, 256)
        elif self.loop_detect == 'query':
            self.desc_dim = cfg.localizer.desc_dim
            self.num_layers = cfg.localizer.num_layers
            self.num_heads = cfg.localizer.num_heads
            self.place_transformer = TransformerPlaceDecoder(self.input_size[0], self.num_heads, self.num_layers, self.desc_dim, self.input_size[1] * self.input_size[2])
            
        # get local descs
        if self.with_local_feats:
            if self.pose_estimate == 'match':
                self.match = True
            elif self.pose_estimate == 'regress':
                self.local_conv_reg_0 = copy.deepcopy(self.layer2)
                self.local_conv_reg_1 = nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                )
            elif self.pose_estimate == 'polar':
                pass

    def rotate_bev_feats(self, feature, yaw):
        B, D, _, _ = feature.shape
        rotated_feature = torch.zeros_like(feature)
        for b in range(B):
            rotated_feature[b] = rotate(feature[b], float(yaw[b]), torchvision.transforms.InterpolationMode.BILINEAR, fill=1)
        return rotated_feature

    def forward_fft(self, input):
        median_output = torch.fft.fft2(input, norm="ortho")
        output = torch.sqrt(median_output.real ** 2 + median_output.imag ** 2 + 1e-15)
        output = fftshift2d(output)
        return output, median_output

    def cartesian_bev_to_polar_bev(self, bev, with_fft=False):
        out_h = self.radius
        out_w = self.theta
        B = bev.shape[0]
        new_h = torch.linspace(0, 1, out_h).view(-1, 1).repeat(1, out_w) #(40, 120)
        new_w = math.pi * torch.linspace(0, 2, out_w).repeat(out_h, 1) # (40, 120)
        grid_xy = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2) #(40, 120, 2)
        new_grid = grid_xy.clone()
        new_grid[...,0] = grid_xy[...,0] * torch.cos(grid_xy[...,1])
        new_grid[...,1] = grid_xy[...,0] * torch.sin(grid_xy[...,1])
        new_grid = new_grid.unsqueeze(0).cuda().repeat(B,1,1,1)
        polar_img_bev = F.grid_sample(bev, new_grid, align_corners=False)# (B, D, theta, radius)
        if with_fft:
            polar_img_bev = torch.fft.fft2(polar_img_bev, norm="ortho")
            polar_img_bev  = torch.sqrt(polar_img_bev.real ** 2 + polar_img_bev.imag ** 2 + 1e-15)
        return polar_img_bev

    def forward(self, bev, ground_bev=None, ego_points=None):
        # bev: (b, d, z, x)-(1, 256, 100, 100)
        out_dict = {}
        out_dict['bev'] = bev

        # Get global descriptors from  bev feats
        if self.loop_detect == 'distance':
            if self.training and self.rotate_aug and random.random() > 0.5:
                sample_yaw = (torch.rand(bev.shape[0], 1) - 0.5) * 360
                bev = self.rotate_bev_feats(bev, sample_yaw)
            global_bev_desc = self.layer2(bev)
            global_bev_desc = self.pooling(global_bev_desc).squeeze(-1).squeeze(-1)
        elif self.loop_detect == 'query':
            global_bev_desc = self.place_transformer(bev.flatten(2, 3).transpose(2, 1))
        elif self.loop_detect == 'pdistance':
            if self.training and self.rotate_aug and random.random() > 0.5:
                sample_yaw = (torch.rand(bev.shape[0], 1) - 0.5) * 360
                polar_bev = self.cartesian_bev_to_polar_bev(self.rotate_bev_feats(bev, sample_yaw))
            else:
                polar_bev = self.cartesian_bev_to_polar_bev(bev)
            polar_bev = self.layer2(polar_bev)
            global_bev_desc = self.pooling(polar_bev).squeeze(-1).squeeze(-1)
        elif self.loop_detect == 'disco':
            # t1 = time.time()
            if self.training and self.rotate_aug and random.random() > 0.5:
                sample_yaw = (torch.rand(bev.shape[0], 1) - 0.5) * 360
                x = self.cartesian_bev_to_polar_bev(self.rotate_bev_feats(bev, sample_yaw))
            else:
                x = self.cartesian_bev_to_polar_bev(bev)
            x = self.disco_conv(x)
            x, fourier_spectrum = self.forward_fft(x)
            x = x[:,:, (self.radius//2 - 12):(self.radius//2 + 12), (self.theta//2 - 12):(self.theta//2 + 12)].squeeze(1)
            global_bev_desc = x.flatten(1, -1)
            # t2 = time.time()
            # print('GD', t2 - t1)
        elif self.loop_detect == 'netvlad':
            x = self.cartesian_bev_to_polar_bev(bev)
            # x = x.permute(0,2,3,1).reshape(bev.shape[0], -1, bev.shape[1])
            x = self.netvlad(x)
            global_bev_desc = self.fc_out(x)

        if self.with_normalize:
            global_bev_desc = F.normalize(global_bev_desc, p=2, dim=1)
        out_dict['loop_desc'] = global_bev_desc

        # Get local bev descriptors from bev feats
        if ground_bev is not None:
            local_bev = ground_bev
        else:
            local_bev = bev.clone()
        if self.with_local_feats:
            if self.pose_estimate == 'regress':
                local_bev_descs = self.local_conv_reg_0(local_bev) #(b, 512, x', y')
                local_bev_descs = self.local_conv_reg_1(local_bev_descs) 
            elif self.pose_estimate == 'match':
                local_bev_descs = local_bev
                out_dict['ego_points'] = ego_points
            elif self.pose_estimate == 'query':
                local_bev_descs = local_bev
            elif self.pose_estimate == 'polar':
                local_bev_descs = local_bev
            out_dict['pose_desc'] = local_bev_descs
        return out_dict

class Bev_desc_decoder_v0(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layer

        resnet = torchvision.models.resnet101(pretrained=False)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, bev):
        # bev: (b, d, z, x)-(1, 256, 100, 100)
        # Get global descriptor
        bev = self.layer2(bev)
        bev = self.layer3(bev)
        desc = self.pooling(bev).squeeze(-1).squeeze(-1)
        desc = F.normalize(desc, p=2, dim=1)
        return {'loop_desc': desc}