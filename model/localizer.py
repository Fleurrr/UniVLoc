import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional
from torchvision.transforms.functional import rotate

from model.plugin import MLP
from model.transformer.vanilla_transformer import TransformerLayer
from model.attention import VanillaSelfAttention, VanillaCrossAttention
from model.transformer.rpe_transformer import RPETransformerLayer
from model.transformer.lrpe_transformer import LRPETransformerLayer
from model.transformer.rpe_radius_transformer import RPERadiusTransformerLayer
from model.transformer.positional_embedding import RelativeAngleEmbedding, RelativeDistanceEmbedding, RelativePositionEmbedding
from model.transformer.pose_transformer import TransformerPoseDecoder, TransformerPoseDecoderLayer

from model.registration.point_matching import PointMatching
from model.registration.matching import get_point_correspondences
from model.registration.point_target_generator import PointTargetGenerator
from model.registration.three_dof_pose_estimator import ThreeDoFPoseEstimator
from model.registration.yaw_searcher import YawSearcher
from model.registration.xy_searcher import XYSearcher
from model.registration.matching import pairwise_distance
from model.registration.transformation import  get_transform_from_rotation_translation

import numpy as np
import random
import heapq
import time
import cv2
import pdb

class BEVBasicPredictor(nn.Module):
    def __init__(
        self, inp_feat_dim=128, feat_dim=128, num_layers=4, out_dim=1):
        super().__init__()
        self.inp_feat_dim = inp_feat_dim
        self.feat_dim = feat_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.predictor = nn.Sequential(
            nn.Conv2d(self.inp_feat_dim, self.feat_dim, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(self.feat_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim // 2, kernel_size=3, stride=3, padding=0) ,
            nn.BatchNorm2d(self.feat_dim // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.fc = MLP(15488, 3872, 2, 3)
        # self.fc = MLP(3200, 800, 2, 3)
    def forward(self, desc_1, desc_2):
        # (b, 512, x', y')
        x = torch.cat([desc_1, desc_2], dim=1) #(b, 512, x' ,y')
        x = self.predictor(x).flatten(1, -1)
        x = self.fc(x)
        return x

class Bev_localizer(nn.Module):
    def __init__(self, cfg, training=True, with_neg=True, with_att=True):
        super().__init__()
        self.with_local_feats = cfg.model.with_local_feats
        self.training = training
        self.loop_thresh = cfg.localizer.loop_thresh
        self.pos_radius = cfg.localizer.pos_radius
        self.X, self.Z = cfg.model.X, cfg.model.Z
        self.grid_size = cfg.model.grid_size
        self.feature_dim = cfg.model.output_dim

        # pose estimator
        self.pose_estimate = cfg.localizer.pose_estimate
        if self.with_local_feats:
            if self.pose_estimate == 'regress': # regress, match
                self.pose_dim = cfg.localizer.pose_dim
                self.pose_estimator_reg_three_dof = BEVBasicPredictor(cfg.model.output_dim * 2, cfg.model.output_dim, out_dim=self.pose_dim)
            elif self.pose_estimate == 'match':              
                self.pose_blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
                self.num_heads = cfg.localizer.num_heads
                self.sigma_a = cfg.model.sigma_a
                self.sigma_d = cfg.model.sigma_d
                self.use_mask = cfg.localizer.use_mask
                self.GPE = RelativePositionEmbedding(self.feature_dim, self.sigma_d, self.sigma_a)
                pose_layers = []
                for pose_block in self.pose_blocks:
                    if pose_block == 'self':
                        pose_layers.append(RPETransformerLayer(self.feature_dim, self.num_heads, dropout=None, activation_fn='ReLU'))
                    else:
                        pose_layers.append(TransformerLayer(self.feature_dim, self.num_heads, dropout=None, activation_fn='ReLU'))
                self.pose_layers = nn.ModuleList(pose_layers)       
                self.coarse_matching = PointMatching(cfg.localizer.num_correspondences, cfg.localizer.dual_normalization)
                self.coarse_target = PointTargetGenerator(cfg.localizer.num_targets, cfg.localizer.distance_threshold)
                self.pose_estimator_three_dof = ThreeDoFPoseEstimator(return_transform=True)
            elif self.pose_estimate == 'query':
                self.pose_dim = cfg.localizer.pose_dim
                self.num_layers = cfg.localizer.num_layers
                self.num_heads = cfg.localizer.num_heads
                self.pose_transformer = TransformerPoseDecoder(self.feature_dim, self.num_heads, self.num_layers, self.pose_dim, self.X * self.Z)
            elif self.pose_estimate == 'polar':
                self.theta = cfg.model.theta
                self.radius = cfg.model.radius
                self.num_heads = cfg.localizer.num_heads
                self.num_layers = cfg.localizer.num_layers
                self.geo_emb = cfg.localizer.geo_emb
                self.estimate_translation = cfg.localizer.estimate_translation
                self.sigma_d = cfg.model.sigma_d
                self.with_att = with_att
                self.with_neg = with_neg
                if self.with_att:
                    self.radius_att = RPERadiusTransformerLayer(self.feature_dim, self.num_heads, dropout=None, activation_fn='ReLU')
                    self.ADE = RelativeDistanceEmbedding(self.feature_dim, self.sigma_d, self.X // (2 * self.radius))
                self.radius_mlp = MLP(self.radius, self.radius // 2, 1, self.num_layers)

                self.theta_blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
                self.sigma_a = cfg.model.sigma_a
                theta_layers = []
                for theta_block in self.theta_blocks:
                    if theta_block == 'self':
                        theta_layers.append(RPETransformerLayer(self.feature_dim, self.num_heads, dropout=None, activation_fn='ReLU'))
                    elif theta_block == 'cross':
                        theta_layers.append(TransformerLayer(self.feature_dim, self.num_heads, dropout=None, activation_fn='ReLU'))
                self.theta_layers = nn.ModuleList(theta_layers)
                self.ARE = RelativeAngleEmbedding(self.feature_dim, self.sigma_a)

                if self.estimate_translation:
                    self.xy_blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
                    self.sigma_d = cfg.model.sigma_d
                    xy_layers = []
                    for xy_block in self.xy_blocks:
                        if xy_block == 'self':
                            xy_layers.append(RPETransformerLayer(self.feature_dim, self.num_heads, dropout=None, activation_fn='ReLU'))
                        elif xy_block == 'cross':
                            xy_layers.append(TransformerLayer(self.feature_dim, self.num_heads, dropout=None, activation_fn='ReLU'))
                    self.xy_layers = nn.ModuleList(xy_layers)
                    self.ADE = RelativeDistanceEmbedding(self.feature_dim, self.sigma_d, self.grid_size)   
                
                # searcher
                self.yaw_searcher = YawSearcher()
                self.searching = cfg.localizer.searching
                # self.searching =False
                if self.estimate_translation:
                    self.xy_searcher = XYSearcher()
                # self.translation_estimator = BEVBasicPredictor(self.feature_dim * 2, self.feature_dim, out_dim=2)

        # loop detector
        self.loop_detect = cfg.localizer.loop_detect
        self.loop_interact = cfg.localizer.loop_interact
        if self.loop_detect == 'score': # distance, 
            self.loop_detector = BasicPredictor(cfg.model.output_dim * 2, cfg.model.out_dim, out_dim=1)
        elif self.loop_detect == 'distance' and self.loop_interact == True:
                self.loop_blocks = ['cross', 'self', 'cross', 'self']
                loop_layers = []
                for loop_block in self.loop_blocks:
                    if loop_block == 'self':
                        loop_layers.append(LoopSelfAttention(feat_dim = self.feature_dim * 2))
                    else:
                        loop_layers.append(LoopCrossAttention(feat_dim = self.feature_dim * 2))
                self.loop_layers = nn.ModuleList(loop_layers)

    def post_process_loop_desc(self, loop_desc):
        return loop_desc

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

    def rotate_bev_feats(self, feature, yaw):
        B, D, _, _ = feature.shape
        rotated_feature = torch.ones_like(feature)
        for b in range(B):
            rotated_feature[b] = rotate(feature[b], float(yaw[b]), torchvision.transforms.InterpolationMode.BILINEAR, fill=0)
        return rotated_feature

    def _transalte_bev_feats(self, feature, step, dim=1):
        D, X, Y = feature.size()
        if step >= 0:
            int_step = int(step)
            alpha = step - int_step
            feature = (1 - alpha) * feature.roll(-int_step, dim) + alpha * feature.roll(-int_step - 1, dim)
            if dim == 1:
                feature[:, X - math.ceil(step):, :] = 1.
            elif dim == 2:
                feature[:, :, Y - math.ceil(step):] = 1.
        else:
            step = abs(step)
            int_step = int(step)
            alpha = step - int_step
            feature = (1 - alpha) * feature.roll(int_step, dim) + alpha * feature.roll(int_step + 1, dim)
            if dim == 1:
                feature[:, :math.ceil(step), :] = 1.
            elif dim == 2:
                feature[:, :, :math.ceil(step)] = 1.
        return feature

    def translate_bev_feats(self, feature, t):
        B, D, _, _ = feature.shape
        translated_feature = torch.ones_like(feature)
        shifts = t / self.grid_size
        for b in range(B):
            step_x, step_y = shifts[b, 0], shifts[b, 1]
            x_shifted_feature = self._transalte_bev_feats(feature[b], step_x, 1)
            xy_shifted_feature = self._transalte_bev_feats(x_shifted_feature, step_y, 2)
            translated_feature[b] = xy_shifted_feature
        return translated_feature

    def useful_t(self, t):
        if torch.any(t < 0.05):
            return False
        return True

    def disturbate_t(self, t):
        while(1):
            dist_t = torch.rand(t.shape[0], 2) * 2
            if self.useful_t(dist_t):
                break
        sign = torch.randint(0, 2, (t.shape[0], 2)) * 2 - 1
        dist_t  = dist_t * sign
        return dist_t.cuda() + t, dist_t

    def _optimal_pose_searching(self, qry_yaw_desc, pos_yaw_desc, d_embedding=None, a_embedding=None, prior_yaw=None):
        # t1 = time.time()
        qry_polar_desc = self.cartesian_bev_to_polar_bev(qry_yaw_desc)
        pos_polar_desc = self.cartesian_bev_to_polar_bev(pos_yaw_desc)
        B = qry_polar_desc.shape[0]
        # t2 = time.time()
        # print('to polar bev:', t2 - t1)
        if self.with_att:
            qry_polar_desc = qry_polar_desc.permute(0, 3, 2, 1)
            qry_polar_desc = self.radius_att(qry_polar_desc, qry_polar_desc, d_embedding)[0].permute(0, 3, 2, 1)
            pos_polar_desc = pos_polar_desc.permute(0, 3, 2, 1)
            pos_polar_desc = self.radius_att(pos_polar_desc, pos_polar_desc, d_embedding)[0].permute(0, 3, 2, 1)
        qry_polar_feats = self.radius_mlp(qry_polar_desc.permute(0, 3, 1, 2)).squeeze(-1) #(B, D, radius, theta) -> (B, theta, D)
        pos_polar_feats = self.radius_mlp(pos_polar_desc.permute(0, 3, 1, 2)).squeeze(-1)
        for i, block in enumerate(self.theta_blocks):
            if block == 'self':
                qry_polar_feats, _ = self.theta_layers[i](qry_polar_feats, qry_polar_feats, a_embedding)
                pos_polar_feats, _ = self.theta_layers[i](pos_polar_feats, pos_polar_feats, a_embedding)
            elif block == 'cross':
                qry_polar_feats, _ = self.theta_layers[i](qry_polar_feats, pos_polar_feats)
                pos_polar_feats, _ = self.theta_layers[i](pos_polar_feats, qry_polar_feats)

        # normalize polar descriptors
        qry_polar_feats = F.normalize(qry_polar_feats, p=2, dim=2)
        pos_polar_feats = F.normalize(pos_polar_feats, p=2, dim=2)
        # t3 = time.time()
        # print('LD:', t3 - t1)
        # search for optimizal minimum yaw cost
        if prior_yaw is not None:
            prior_yaw = int(prior_yaw / (360 / self.theta))
        optimal_step, min_cost = self.yaw_searcher(qry_polar_feats[0, ...], pos_polar_feats[0, ...], max_step=self.theta // 2, step_size=1.0, prior_yaw=prior_yaw, search_range=5)
        optimal_yaw = optimal_step * (360 / self.theta)
        # t4 = time.time()
        # print('searching:', t4 - t3)
        return optimal_yaw, min_cost
    
    def optimal_pose_searching_grid(self, qry_yaw_desc, pos_yaw_desc, prior_yaw=None, bound=2, grid=20, terminate_cost=0.0, dump_img=True):
        optimal_x, optimal_y, optimal_t = None , None, None
        optimal_cost = float('inf')
        bounds = [(-bound, bound), (-bound, bound)]
        candidate_xy = []
        candidate_cost = []

        B = qry_yaw_desc.shape[0]
        d_embedding = self.ADE(self.radius).unsqueeze(0).unsqueeze(0).repeat(B, self.theta, 1, 1, 1)
        a_embedding = self.ARE(self.theta).unsqueeze(0).repeat(B, 1, 1, 1)
        if dump_img:
            value = np.ones((grid, grid))
        for idx in range(grid):
            for idy in range(grid):
                # t1 = time.time()
                x, y = idx * (bounds[0][1] - bounds[0][0]) / grid - bound, idy * (bounds[0][1] - bounds[0][0]) / grid - bound
                t_proposal =  torch.Tensor([x, y]).unsqueeze(0)
                qry_yaw_desc_translated = self.translate_bev_feats(qry_yaw_desc, t_proposal)
                local_yaw, local_cost = self._optimal_pose_searching(qry_yaw_desc_translated, pos_yaw_desc, d_embedding, a_embedding, prior_yaw=prior_yaw)
                if dump_img:
                    value[idx][idy] = local_cost
                if local_cost < optimal_cost:
                    optimal_x, optimal_y = x, y 
                    optimal_cost = local_cost
                    optimal_yaw = local_yaw
                    optimal_t = t_proposal
                    if local_cost < terminate_cost:
                        break
                candidate_xy.append(t_proposal)
                candidate_cost.append(optimal_cost)
        if dump_img:
            value  = value - value.min()
            value  = (value / value.max()) * 255
            cv2.imwrite('search.png', value)
        return optimal_yaw, optimal_t, optimal_cost

    def optimal_pose_searching(self, qry_yaw_desc, pos_yaw_desc, prior_yaw=None, bound=2, split=4, split_stage=1, choose=3, random_stage=25):
        optimal_x, optimal_y, optimal_t = None , None, None
        optimal_cost = float('inf')
        bounds = list([[-bound, bound], [-bound, bound]])

        B = qry_yaw_desc.shape[0]
        if self.with_att:
            d_embedding = self.ADE(self.radius).unsqueeze(0).unsqueeze(0).repeat(B, self.theta, 1, 1, 1)
        else:
            d_embedding = None

        if self.geo_emb:
            a_embedding = self.ARE(self.theta).unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            a_embedding = None
        
        for _ in range(split_stage): # coarse stage
            x_step = (bounds[0][1] - bounds[0][0]) / split
            y_step = (bounds[1][1] - bounds[1][0]) / split
            candidate_x = []
            candidate_y = []
            candidate_cost = []
            for idx in range(split + 1):
                for idy in range(split + 1):
                    x, y = x_step * idx + bounds[0][0], y_step * idy + bounds[1][0]
                    # t8 = time.time()
                    t_proposal =  torch.Tensor([x, y]).unsqueeze(0)
                    qry_yaw_desc_translated = self.translate_bev_feats(qry_yaw_desc, t_proposal)
                    local_yaw, local_cost = self._optimal_pose_searching(qry_yaw_desc_translated, pos_yaw_desc, d_embedding, a_embedding, prior_yaw=prior_yaw)
                    # t9 = time.time()
                    # print('rr', t9 - t8)
                    candidate_x.append(x)
                    candidate_y.append(y)
                    candidate_cost.append(local_cost)
                    if local_cost < optimal_cost:
                        optimal_x, optimal_y = x, y 
                        optimal_cost = local_cost
                        optimal_yaw = local_yaw
                        optimal_t = t_proposal

            min_cost_index = list(map(candidate_cost.index, heapq.nsmallest(choose, candidate_cost)))
            min_x_list, min_y_list = np.array(candidate_x)[min_cost_index].tolist(), np.array(candidate_y)[min_cost_index].tolist()
            min_x_bound, max_x_bound = min(min_x_list), max(min_x_list)
            min_y_bound, max_y_bound = min(min_y_list), max(min_y_list)
            bounds[0][0], bounds[0][1] = min_x_bound, max_x_bound
            bounds[1][0], bounds[1][1] = min_y_bound, max_y_bound

        for _ in range(random_stage): # fine stage
            x, y = random.uniform(bounds[0][0], bounds[0][1]), random.uniform(bounds[1][0], bounds[1][1])
            t_proposal =  torch.Tensor([x, y]).unsqueeze(0)
            qry_yaw_desc_translated = self.translate_bev_feats(qry_yaw_desc, t_proposal)
            local_yaw, local_cost = self._optimal_pose_searching(qry_yaw_desc_translated, pos_yaw_desc, d_embedding, a_embedding, prior_yaw=prior_yaw)
            candidate_x.append(x)
            candidate_y.append(y)
            candidate_cost.append(optimal_cost)
            if local_cost < optimal_cost:
                optimal_x, optimal_y = x, y 
                optimal_cost = local_cost
                optimal_yaw = local_yaw
                optimal_t = t_proposal            

        return optimal_yaw, optimal_t, optimal_cost

    def matching_based_loop_detection(self, qry_yaw_desc, pos_yaw_desc, loop_thresh=0.1, count_thresh=10):
        qry_polar_desc = self.cartesian_bev_to_polar_bev(qry_yaw_desc)
        pos_polar_desc = self.cartesian_bev_to_polar_bev(pos_yaw_desc)
        B = qry_polar_desc.shape[0]
        if self.with_att:
            d_embedding = self.ADE(self.radius).unsqueeze(0).unsqueeze(0).repeat(B, self.theta, 1, 1, 1)
            qry_polar_desc = qry_polar_desc.permute(0, 3, 2, 1)
            qry_polar_desc = self.radius_att(qry_polar_desc, qry_polar_desc, d_embedding)[0].permute(0, 3, 2, 1)
            pos_polar_desc = pos_polar_desc.permute(0, 3, 2, 1)
            pos_polar_desc = self.radius_att(pos_polar_desc, pos_polar_desc, d_embedding)[0].permute(0, 3, 2, 1)
        qry_polar_feats = self.radius_mlp(qry_polar_desc.permute(0, 3, 1, 2)).squeeze(-1) #(B, D, radius, theta) -> (B, theta, D)
        pos_polar_feats = self.radius_mlp(pos_polar_desc.permute(0, 3, 1, 2)).squeeze(-1)
        feat_dists = torch.sqrt(pairwise_distance(qry_polar_feats, pos_polar_feats, normalized=False))
        return feat_dists.mean()
    
    def polar_feature_interact(self, qry_yaw_desc, pos_yaw_desc):
        qry_polar_desc = self.cartesian_bev_to_polar_bev(qry_yaw_desc)
        pos_polar_desc = self.cartesian_bev_to_polar_bev(pos_yaw_desc)
        # (B, D, X, Y) -> (B, D, theta, radius)
        B = qry_polar_desc.shape[0]
        if self.with_att:
            d_embedding = self.ADE(self.radius).unsqueeze(0).unsqueeze(0).repeat(B, self.theta, 1, 1, 1)
            qry_polar_desc = qry_polar_desc.permute(0, 3, 2, 1)
            qry_polar_desc = self.radius_att(qry_polar_desc, qry_polar_desc, d_embedding)[0].permute(0, 3, 2, 1)
            pos_polar_desc = pos_polar_desc.permute(0, 3, 2, 1)
            pos_polar_desc = self.radius_att(pos_polar_desc, pos_polar_desc, d_embedding)[0].permute(0, 3, 2, 1)
        qry_polar_feats = self.radius_mlp(qry_polar_desc.permute(0, 3, 1, 2)).squeeze(-1) #(B, D, radius, theta) -> (B, theta, D)
        pos_polar_feats = self.radius_mlp(pos_polar_desc.permute(0, 3, 1, 2)).squeeze(-1)
        if self.geo_emb:
            a_embedding = self.ARE(self.theta).unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            a_embedding = None
        for i, block in enumerate(self.theta_blocks):
            if block == 'self':
                qry_polar_feats, _ = self.theta_layers[i](qry_polar_feats, qry_polar_feats, a_embedding)
                pos_polar_feats, _ = self.theta_layers[i](pos_polar_feats, pos_polar_feats, a_embedding)
            elif block == 'cross':
                qry_polar_feats, _ = self.theta_layers[i](qry_polar_feats, pos_polar_feats)
                pos_polar_feats, _ = self.theta_layers[i](pos_polar_feats, qry_polar_feats)

        # normalize polar descriptors
        qry_polar_feats = F.normalize(qry_polar_feats, p=2, dim=2)
        pos_polar_feats = F.normalize(pos_polar_feats, p=2, dim=2)
        return qry_polar_feats, pos_polar_feats

    def forward(self, input_dict, training=True, loop_detect=False):
        out_dict = {}
        # input_dict in training mode
        # qry_loop_desc, pos_loop_desc, neg_loop_desc, *amb_loop_desc
        # qry_pose_desc, pos_pose_desc, neg_pose_desc, *amb_pose_desc 
        if training:
            # estimate pose
            if self.with_local_feats:
                qry_pose_desc = input_dict['qry_pose_desc']
                pos_pose_desc = input_dict['pos_pose_desc']
                B = pos_pose_desc.shape[0]
                if self.pose_estimate == 'regress':
                    poses = self.pose_estimator_reg_three_dof(qry_pose_desc, pos_pose_desc) # (B, 7)
                    if self.pose_dim == 7:
                        quaternions = poses[:, :4]
                        translations = poses[:, 4:]
                        out_dict['rel_q'] = quaternions
                        out_dict['rel_t'] = translations
                    elif self.pose_dim == 3:
                        yaw = poses[:, :1]
                        translations = poses[:, 1:]
                        out_dict['rel_yaw'] = yaw
                        out_dict['rel_t'] = translations

                elif self.pose_estimate == 'match':
                    qry_ego_points = input_dict['qry_ego_points'].flatten(1, 2) #(b, 100, 100, 3) -> (b, 100*100, 3)
                    pos_ego_points = input_dict['pos_ego_points'].flatten(1, 2) #(b, 100, 100, 3) -> (b, 100*100, 3)
                    qry_pose_desc = qry_pose_desc.flatten(2, 3).transpose(2, 1) # (b, d, 100, 100) -> (b, 100*100, d)
                    pos_pose_desc = pos_pose_desc.flatten(2, 3).transpose(2, 1) # (b, d, 100, 100) -> (b, 100*100, d)

                    # calculate masks
                    if self.use_mask:
                        per_cam_bev_mask = input_dict['bev_mask'][:, 0, :, 0] # (S, X * Y)
                        bev_mask = torch.sum(per_cam_bev_mask, dim=0)
                        bev_mask = torch.where(bev_mask > 0, torch.ones_like(bev_mask), torch.zeros_like(bev_mask))
                        qry_ego_points = qry_ego_points[:, bev_mask == 1, :]
                        pos_ego_points = pos_ego_points[:, bev_mask == 1, :]
                        qry_pose_desc = qry_pose_desc[:, bev_mask == 1, :]
                        pos_pose_desc = pos_pose_desc[:, bev_mask == 1, :]

                        # _, N, _ = qry_ego_points.size()
                        # indices = torch.randperm(N)[:self.X * self.X // 4]
                        # qry_ego_points = qry_ego_points[:, indices, :]
                        # pos_ego_points = pos_ego_points[:, indices, :]
                        # qry_pose_desc = qry_pose_desc[:, indices, :]
                        # pos_pose_desc = pos_pose_desc[:, indices, :]                        

                    rel_R = input_dict['rel_R']
                    rel_t = input_dict['rel_t']
                    batch_size = qry_ego_points.shape[0]
                    # self and cross attention between two local bevs
                    # qry_pos_embedding = self.GPE(qry_ego_points)
                    # pos_pos_embedding = self.GPE(pos_ego_points)
                    for i, block in enumerate(self.pose_blocks):
                        if block == 'self':
                            qry_pose_desc, _ = self.pose_layers[i](qry_pose_desc, qry_pose_desc)
                            pos_pose_desc, _ = self.pose_layers[i](pos_pose_desc, pos_pose_desc)
                        elif block == 'cross':
                            qry_pose_desc, _ = self.pose_layers[i](qry_pose_desc, pos_pose_desc)
                            pos_pose_desc, _ = self.pose_layers[i](pos_pose_desc, qry_pose_desc)
                    
                    # normalize local descriptors
                    qry_pose_desc = F.normalize(qry_pose_desc, p=2, dim=2)
                    pos_pose_desc = F.normalize(pos_pose_desc, p=2, dim=2)
                    out_dict['qry_pose_descs'] = qry_pose_desc
                    out_dict['pos_pose_descs'] = pos_pose_desc

                    # get ground-truth corresponding indices
                    rel_T =  get_transform_from_rotation_translation(rel_R, rel_t) #(B, 4, 4)
                    gt_point_corr_indices, gt_point_corr_distances = [], []
                    for idb in range(batch_size):
                        gt_point_corr_indice, gt_point_corr_distance = get_point_correspondences(qry_ego_points[idb, ...], pos_ego_points[idb, ...], rel_T[idb, ...], self.pos_radius)
                        gt_point_corr_indices.append(gt_point_corr_indice)
                        gt_point_corr_distances.append(gt_point_corr_distance)
                    out_dict['gt_point_corr_indices'] = gt_point_corr_indices
                    out_dict['gt_point_corr_distances'] = gt_point_corr_distances
                    
                    # get feature-based matching result
                    with torch.no_grad():
                        qry_point_corr_indices, pos_point_corr_indices, point_corr_scores = [], [], []
                        estimated_transforms = []
                        for idb in range(batch_size):
                            qry_point_corr_indice, pos_point_corr_indice, point_corr_score = self.coarse_matching(
                                qry_pose_desc[idb], pos_pose_desc[idb]
                            )
                            estimated_transform = self.pose_estimator_three_dof(qry_ego_points[idb][qry_point_corr_indice], pos_ego_points[idb][pos_point_corr_indice], point_corr_score)
                            # estimated_transform = self.pose_estimator_three_dof(qry_ego_points[idb][gt_point_corr_indices[idb][:,0]], pos_ego_points[idb][gt_point_corr_indices[idb][:,1]]) # estimate with ground-truth corrs

                            qry_point_corr_indices.append(qry_point_corr_indice)
                            pos_point_corr_indices.append(pos_point_corr_indice)
                            point_corr_scores.append(point_corr_score)
                            estimated_transforms.append(estimated_transform)

                        out_dict['qry_point_corr_indices'] = qry_point_corr_indices
                        out_dict['pos_point_corr_indices'] = pos_point_corr_indices
                        out_dict['point_corr_scores'] = point_corr_scores
                        out_dict['estimated_transforms'] = torch.stack(estimated_transforms)

                elif self.pose_estimate == 'query':
                    qry_pose_desc = qry_pose_desc.flatten(2, 3).transpose(2, 1) # (b, d, 100, 100) -> (b, 100*100, d)
                    pos_pose_desc = pos_pose_desc.flatten(2, 3).transpose(2, 1) # (b, d, 100, 100) -> (b, 100*100, d)
                    poses = self.pose_transformer(qry_pose_desc, pos_pose_desc) # (B, 7)
                    B = qry_pose_desc.shape[0]
                    if self.pose_dim == 7: # first 4 quaternion, last 3 xyz
                        quaternions = poses[:, :4]
                        translations = poses[:, 4:]
                        out_dict['rel_q'] = quaternions
                        out_dict['rel_t'] = translations
                    elif self.pose_dim ==4: # quaternion only
                        out_dict['rel_q'] = poses
                        out_dict['rel_t'] = torch.ones(B, 3).float().cuda()
                    elif self.pose_dim == 3: # first 1 yaw, last 2 xy
                        yaw = poses[:, :1]
                        translations = poses[:, 1:]
                        out_dict['rel_yaw'] = yaw
                        out_dict['rel_t'] = translations 
                    elif self.pose_dim == 2: # xy only
                        out_dict['rel_yaw'] = torch.ones(B, 1).float().cuda()
                        out_dict['rel_t'] = poses
                    elif self.pose_dim == 1: # yaw only
                        out_dict['rel_yaw'] = poses
                        out_dict['rel_t'] = torch.ones(B, 2).float().cuda()

                elif self.pose_estimate == 'polar':
                    qry_yaw_desc = qry_pose_desc
                    pos_yaw_desc = pos_pose_desc

                    # (B, D, theta, radius)
                    yaw_gt = input_dict['rel_yaw']
                    t_gt = input_dict['rel_translation']
                    translated_qry_yaw_desc = self.translate_bev_feats(qry_yaw_desc, t_gt)
                    qry_polar_feats, pos_polar_feats = self.polar_feature_interact(translated_qry_yaw_desc, pos_yaw_desc)
                    out_dict['qry_pose_descs'] = qry_polar_feats
                    out_dict['pos_pose_descs'] = pos_polar_feats
                    out_dict['rel_translation'] = t_gt  
                    if self.with_neg:
                        t_sample, dist_t = self.disturbate_t(t_gt)
                        neg_yaw_desc = self.translate_bev_feats(qry_yaw_desc, t_sample)
                        neg_pos_polar_feats, pos_neg_polar_feats = self.polar_feature_interact(neg_yaw_desc, pos_yaw_desc)
                        out_dict['neg_pos_pose_descs'] = neg_pos_polar_feats
                        out_dict['pos_neg_pose_descs'] = pos_neg_polar_feats
                        out_dict['dist_t'] = dist_t                 

                    # search for optimal minimum yaw cost
                    rel_yaws = []
                    min_costs = []
                    with torch.no_grad():
                        for idb in range(B):
                            optimal_step, min_cost = self.yaw_searcher(qry_polar_feats[idb, ...], pos_polar_feats[idb, ...], max_step=self.theta // 2)
                            optimal_yaw = optimal_step * (360 / self.theta)
                            rel_yaws.append(optimal_yaw)
                            min_costs.append(min_cost)
                        out_dict['rel_yaw'] = torch.from_numpy(np.stack(rel_yaws)).unsqueeze(1).float()
                        out_dict['min_costs'] = torch.from_numpy(np.stack(min_costs)).unsqueeze(1).float()

                    # regress translation
                    # yaw_gt = input_dict['rel_yaw']
                    # pos_rotated_pose_desc = self.rotate_bev_feats(pos_pose_desc, yaw_gt)
                    # out_dict['rel_t'] = self.translation_estimator(qry_pose_desc, pos_rotated_pose_desc)

                    # for translation
                    if not self.estimate_translation:
                        out_dict['rel_t'] = torch.ones(B, 2).float().cuda()
                    else:
                        yaw_gt = input_dict['rel_yaw']
                        pos_rotated_pose_desc = self.rotate_bev_feats(pos_pose_desc, yaw_gt)
                        qry_x_feats = self.xy_compressor(qry_pose_desc.permute(0, 3, 2, 1)).squeeze(1) #(B, D, X, Y) -> (B, X, D)
                        pos_x_feats = self.xy_compressor(pos_rotated_pose_desc.permute(0, 3, 2, 1)).squeeze(1)  #(B, D, X, Y) -> (B, X, D)
                        qry_y_feats = self.xy_compressor(qry_pose_desc.permute(0, 2, 3, 1)).squeeze(1) #(B, D, X, Y) -> (B, Y, D)
                        pos_y_feats = self.xy_compressor(pos_rotated_pose_desc.permute(0, 2, 3, 1)).squeeze(1)  #(B, D, X, Y) -> (B, Y, D)
                        if self.geo_emb:
                            d_embedding = self.ADE(self.X).unsqueeze(0).repeat(B, 1, 1, 1)
                        else:
                            d_embedding = None
                        for i, block in enumerate(self.xy_blocks):
                            if block == 'self':
                                qry_x_feats, _ = self.xy_layers[i](qry_x_feats, qry_x_feats, d_embedding)
                                qry_y_feats, _ = self.xy_layers[i](qry_y_feats, qry_y_feats, d_embedding)
                                pos_x_feats, _ = self.xy_layers[i](pos_x_feats, pos_x_feats, d_embedding)
                                pos_y_feats, _ = self.xy_layers[i](pos_y_feats, pos_y_feats, d_embedding)
                            elif block == 'cross':
                                qry_x_feats, _ = self.xy_layers[i](qry_x_feats, pos_x_feats)
                                pos_x_feats, _ = self.xy_layers[i](pos_x_feats, qry_x_feats)
                                qry_y_feats, _ = self.xy_layers[i](qry_y_feats, pos_y_feats)
                                pos_y_feats, _ = self.xy_layers[i](pos_y_feats, qry_y_feats)

                        qry_x_feats = F.normalize(qry_x_feats, p=2, dim=2)
                        pos_x_feats = F.normalize(pos_x_feats, p=2, dim=2)
                        qry_y_feats = F.normalize(qry_y_feats, p=2, dim=2)
                        pos_y_feats = F.normalize(pos_y_feats, p=2, dim=2)
                        out_dict['qry_x_descs'] = qry_x_feats
                        out_dict['pos_x_descs'] = pos_x_feats
                        out_dict['qry_y_descs'] = qry_y_feats
                        out_dict['pos_y_descs'] = pos_y_feats

                        # search for optimizal minimum xy cost
                        rel_xs, rel_ys = [], []
                        with torch.no_grad():
                            for idb in range(B):
                                optimal_x_step, _ = self.xy_searcher(qry_x_feats[idb, ...], pos_x_feats[idb, ...], max_step=self.X // 5)                       
                                optimal_x = optimal_x_step * self.grid_size
                                rel_xs.append(optimal_x)
                                optimal_y_step, _ = self.xy_searcher(qry_y_feats[idb, ...], pos_y_feats[idb, ...], max_step=self.X // 5)                       
                                optimal_y = optimal_y_step * self.grid_size
                                rel_ys.append(optimal_y)
                            out_dict['rel_x'] = torch.from_numpy(np.stack(rel_xs)).unsqueeze(1).float()
                            out_dict['rel_y'] = torch.from_numpy(np.stack(rel_ys)).unsqueeze(1).float()
                            out_dict['rel_t'] = torch.cat([out_dict['rel_x'], out_dict['rel_y']], 1)

            # loop closure detection
            qry_loop_desc = self.post_process_loop_desc(input_dict['qry_loop_desc'])
            pos_loop_desc = self.post_process_loop_desc(input_dict['pos_loop_desc'])                   
            neg_loop_descs = []
            for neg_loop_desc in input_dict['neg_loop_desc']:
                neg_loop_desc = self.post_process_loop_desc(neg_loop_desc)
                neg_loop_descs.append(neg_loop_desc )
            out_dict['qry_loop_desc'] = qry_loop_desc
            out_dict['pos_loop_desc'] = pos_loop_desc
            out_dict['neg_loop_desc'] = neg_loop_descs
            if 'amb_loop_desc' in input_dict:
                out_dict['amb_loop_desc'] = self.post_process_loop_desc(input_dict['amb_loop_desc'])
        
        # for evaluating
        else:
            if self.with_local_feats:
                qry_pose_desc = input_dict['pose_desc_1']
                pos_pose_desc = input_dict['pose_desc_2']
                if self.pose_estimate == 'regress':
                    poses = self.pose_estimator_reg_three_dof(qry_pose_desc, pos_pose_desc) # (B, 7)
                    if self.pose_dim == 7:
                        quaternions = poses[:, :4]
                        translations = poses[:, 4:]
                        out_dict['rel_q'] = quaternions
                        out_dict['rel_t'] = translations
                    elif self.pose_dim == 3:
                        yaw = poses[:, :1]
                        translations = poses[:, 1:]
                        out_dict['rel_yaw'] = yaw
                        out_dict['rel_t'] = translations
                
                elif self.pose_estimate == 'query':
                    qry_pose_desc = qry_pose_desc.flatten(2, 3).transpose(2, 1) # (b, d, 100, 100) -> (b, 100*100, d)
                    pos_pose_desc = pos_pose_desc.flatten(2, 3).transpose(2, 1) # (b, d, 100, 100) -> (b, 100*100, d)
                    poses = self.pose_transformer(qry_pose_desc, pos_pose_desc) # (B, P)
                    B = qry_pose_desc.shape[0]
                    if self.pose_dim == 7: # first 4 quaternion, last 3 xyz
                        quaternions = poses[:, :4]
                        translations = poses[:, 4:]
                        out_dict['rel_q'] = quaternions
                        out_dict['rel_t'] = translations
                    elif self.pose_dim ==4: # quaternion only
                        out_dict['rel_q'] = poses
                        out_dict['rel_t'] = torch.ones(B, 3).float().cuda()
                    elif self.pose_dim == 3: # first 1 yaw, last 2 xy
                        yaw = poses[:, :1]
                        translations = poses[:, 1:]
                        out_dict['rel_yaw'] = yaw
                        out_dict['rel_t'] = translations 
                    elif self.pose_dim == 2: # xy only
                        out_dict['rel_yaw'] = torch.ones(B, 1).float().cuda()
                        out_dict['rel_t'] = poses
                    elif self.pose_dim == 1: # yaw only
                        out_dict['rel_yaw'] = poses
                        out_dict['rel_t'] = torch.ones(B, 2).float().cuda()

                elif self.pose_estimate == 'match':
                    qry_ego_points = input_dict['ego_points_1'].flatten(1, 2)
                    pos_ego_points = input_dict['ego_points_2'].flatten(1, 2)
                    qry_pose_desc = qry_pose_desc.flatten(2, 3).transpose(2, 1) # (b, d, 100, 100) -> (b, 100*100, d)
                    pos_pose_desc = pos_pose_desc.flatten(2, 3).transpose(2, 1) # (b, d, 100, 100) -> (b, 100*100, d)W
                    # calculate masks
                    if self.use_mask:
                        per_cam_bev_mask = input_dict['bev_mask'][:, 0, :, 0] # (S, X * Y)
                        bev_mask = torch.sum(per_cam_bev_mask, dim=0)
                        bev_mask = torch.where(bev_mask > 0, torch.ones_like(bev_mask), torch.zeros_like(bev_mask))
                        qry_ego_points = qry_ego_points[:, bev_mask == 1, :]
                        pos_ego_points = pos_ego_points[:, bev_mask == 1, :]
                        qry_pose_desc = qry_pose_desc[:, bev_mask == 1, :]
                        pos_pose_desc = pos_pose_desc[:, bev_mask == 1, :]

                    rel_R = input_dict['rel_R']
                    rel_t = input_dict['rel_t']
                    batch_size = qry_ego_points.shape[0]
                    # self and cross attention between two local bevs
                    # qry_pos_embedding = self.GPE(qry_ego_points)
                    # pos_pos_embedding = self.GPE(pos_ego_points)
                    for i, block in enumerate(self.pose_blocks):
                        if block == 'self':
                            qry_pose_desc, _ = self.pose_layers[i](qry_pose_desc, qry_pose_desc)
                            pos_pose_desc, _ = self.pose_layers[i](pos_pose_desc, pos_pose_desc)
                        elif block == 'cross':
                            qry_pose_desc, _ = self.pose_layers[i](qry_pose_desc, pos_pose_desc)
                            pos_pose_desc, _ = self.pose_layers[i](pos_pose_desc, qry_pose_desc)
                    
                    # normalize local descriptors
                    qry_pose_desc = F.normalize(qry_pose_desc, p=2, dim=2)
                    pos_pose_desc = F.normalize(pos_pose_desc, p=2, dim=2)
                    out_dict['qry_pose_descs'] = qry_pose_desc
                    out_dict['pos_pose_descs'] = pos_pose_desc

                    # get feature-based matching result
                    with torch.no_grad():
                        qry_point_corr_indices, pos_point_corr_indices, point_corr_scores = [], [], []
                        estimated_transforms = []
                        for idb in range(batch_size):
                            qry_point_corr_indice, pos_point_corr_indice, point_corr_score = self.coarse_matching(
                                qry_pose_desc[idb], pos_pose_desc[idb]
                            )
                            estimated_transform = self.pose_estimator_three_dof(qry_ego_points[idb][qry_point_corr_indice], pos_ego_points[idb][pos_point_corr_indice], point_corr_score)
                            # estimated_transform = self.pose_estimator_three_dof(qry_ego_points[idb][gt_point_corr_indices[idb][:,0]], pos_ego_points[idb][gt_point_corr_indices[idb][:,1]]) # estimate with ground-truth corrs
                            estimated_transforms.append(estimated_transform)
                    estimated_transforms = torch.stack(estimated_transforms)
                    out_dict['estimated_transforms'] = estimated_transforms

                elif self.pose_estimate == 'polar':
                    qry_yaw_desc = qry_pose_desc
                    pos_yaw_desc = pos_pose_desc
                    # local_feat_dist = self.matching_based_loop_detection(qry_yaw_desc, pos_yaw_desc)
                    # print('local feature distance:', local_feat_dist.item())
                    
                    if self.searching:
                        # only support batch size = 1
                        if 'prior_yaw' in input_dict.keys():
                            prior_yaw = input_dict['prior_yaw']
                        else:
                            prior_yaw = None
                        optimal_yaw, optimal_t, optimal_cost = self.optimal_pose_searching(qry_yaw_desc, pos_yaw_desc, prior_yaw=prior_yaw)
                        out_dict['rel_t'] = optimal_t.unsqueeze(0)
                        out_dict['rel_yaw'] = optimal_yaw
                        out_dict['min_costs'] = optimal_cost
                    else:
                        # if coarse translation estimate exists
                        if 'rel_t' in input_dict.keys():
                            rel_t_gt = input_dict['rel_t']
                            qry_yaw_desc = self.translate_bev_feats(qry_yaw_desc, rel_t_gt)
                            # out_dict['qry_translated_bev'] = qry_yaw_desc

                        B = qry_yaw_desc.shape[0]
                        qry_polar_feats, pos_polar_feats = self.polar_feature_interact(qry_yaw_desc, pos_yaw_desc)
                        out_dict['qry_pose_descs'] = qry_polar_feats
                        out_dict['pos_pose_descs'] = pos_polar_feats

                        # search for optimizal minimum yaw cost
                        rel_yaws = []
                        min_costs = []
                        with torch.no_grad():
                            for idb in range(B):
                                optimal_step, min_cost = self.yaw_searcher(qry_polar_feats[idb, ...], pos_polar_feats[idb, ...], max_step=self.theta // 2, step_size=1.0, prior_yaw=None, search_range=5)
                                optimal_yaw = optimal_step * (360 / self.theta)
                                rel_yaws.append(optimal_yaw)
                                min_costs.append(min_cost)
                            out_dict['rel_yaw'] = torch.from_numpy(np.stack(rel_yaws)).unsqueeze(1).float()
                            out_dict['min_costs'] = torch.from_numpy(np.stack(min_costs)).unsqueeze(1).float()

                        # for translation
                        out_dict['rel_t'] = [torch.ones(B, 2)]
        return out_dict