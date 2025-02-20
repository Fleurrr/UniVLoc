import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.tensor import tensor_to_array
from utils.geometry import quaternion_to_rotation
from utils.registration import compute_registration_error
from model.registration.matching import pairwise_distance

from loss.circle_loss import WeightedCircleLoss

import pdb

class LossFunction_MTC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.match_loss.positive_margin,
            cfg.match_loss.negative_margin,
            cfg.match_loss.positive_optimal,
            cfg.match_loss.negative_optimal,
            cfg.match_loss.log_scale,
        )
        self.positive_distance = cfg.match_loss.positive_distance        
        self.loss_weight = cfg.loss.pose_loss_weight
    def forward(self, out_dict, data_dict):
        match_loss = 0
        qry_feats = out_dict['qry_pose_descs'] # (B, N, D)
        pos_feats = out_dict['pos_pose_descs'] # (B, N, D)
        gt_point_corr_indices = out_dict['gt_point_corr_indices'] #[]
        gt_point_corr_distances = out_dict['gt_point_corr_distances'] # []
        for idb in range(qry_feats.shape[0]):
            gt_qry_point_corr_indices = gt_point_corr_indices[idb][:, 0] #(M)
            gt_pos_point_corr_indices = gt_point_corr_indices[idb][:, 1] #(M)
            feat_dists = torch.sqrt(pairwise_distance(qry_feats[idb], pos_feats[idb], normalized=True))

            overlaps = torch.zeros_like(feat_dists)
            overlaps[gt_qry_point_corr_indices, gt_pos_point_corr_indices] = (self.positive_distance - gt_point_corr_distances[idb]) / self.positive_distance
            pos_masks = torch.gt(overlaps, 0.2)
            neg_masks = torch.eq(overlaps, 0)
            pos_scales = torch.sqrt(overlaps * pos_masks.float())

            loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)
            if torch.isnan(loss):
                continue
            match_loss += loss
        return {'pose_loss': match_loss * self.loss_weight}

class LossFunction_REG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.s_t = torch.nn.Parameter(torch.Tensor([.0]), requires_grad=True).cuda()
        self.s_q = torch.nn.Parameter(torch.Tensor([.0]), requires_grad=True).cuda()

        self.loss = nn.MSELoss(reduction='mean')
        self.loss_weight = cfg.loss.pose_loss_weight
        self.pose_dim = cfg.localizer.pose_dim

    def forward(self, out_dict, data_dict):
        if self.pose_dim == 7:
            predict_q = out_dict['rel_q']
            predict_t = out_dict['rel_t']
            groundtruth_q = data_dict['relative_quaternion']
            groundtruth_t = data_dict['relative_translation']
        elif self.pose_dim == 4:
            predict_q = out_dict['rel_q']
            groundtruth_q = data_dict['relative_quaternion']
        elif self.pose_dim == 3:
            predict_q = out_dict['rel_yaw']
            predict_t = out_dict['rel_t']
            groundtruth_q = data_dict['relative_euler'][...,2:] / 180.
            groundtruth_t = data_dict['relative_translation'][...,:2]
        elif self.pose_dim == 2:
            predict_t = out_dict['rel_t']
            groundtruth_t = data_dict['relative_translation'][...,:2]
        elif self.pose_dim == 1:
            predict_q = out_dict['rel_yaw']
            groundtruth_q = data_dict['relative_euler'][...,2:] / 180.

        if self.pose_dim in [2, 3, 7]:
            l_t = torch.norm(groundtruth_t - predict_t, dim=1, p=2).mean()
        else:
            l_t = 0
        if self.pose_dim in [4, 7]:
            l_q = torch.norm(F.normalize(groundtruth_q, p=2, dim=1) - F.normalize(predict_q, p=2, dim=1),dim=1, p=2).mean()
        elif self.pose_dim in [1, 3]:
            l_q = self.loss(predict_q, groundtruth_q)
        else:
            l_q = 0
        pose_loss = l_t * torch.exp(-self.s_t) + self.s_t + l_q * torch.exp(-self.s_q) + self.s_q
        return {'pose_loss': pose_loss[0] * self.loss_weight}

class LossFunction_ROT(nn.Module):
    def __init__(self, cfg, trans_weighting=False):
        super().__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.match_loss.positive_margin,
            cfg.match_loss.negative_margin,
            cfg.match_loss.positive_optimal,
            cfg.match_loss.negative_optimal,
            cfg.match_loss.log_scale,
        )
        self.loss_weight = cfg.loss.pose_loss_weight
        self.theta = cfg.model.theta
        self.grid_size = cfg.model.grid_size
        self.estimate_translation = cfg.localizer.estimate_translation

        self.trans_weighting = trans_weighting
        self.tra_loss_weight = 50.
        if self.trans_weighting:
            self.d_weight = 0.02
            w = torch.arange(0, 360, 360 // self.theta)
            self.x_disturbate_weight = torch.abs(torch.cos(w * math.pi/ 180))
            self.y_disturbate_weight = torch.abs(torch.sin(w * math.pi/ 180))
        else:
            self.d_factor = 0.01

    def get_overlaps(self, N, weight, rot=True):    
        # overlaps = torch.zeros(N, N).fill_diagonal_(1).cuda()
        overlaps = torch.zeros(N, N).cuda()
        for i in range(N):
            if i == 0:
                if weight > 0:
                    overlaps[i, i] = 1 - weight
                    overlaps[i, i + 1] =  weight
                elif weight < 0:
                    weight = torch.abs(weight)
                    overlaps[i, i] =  1 - weight
                    if rot:
                        overlaps[i, -1] = weight
            elif i == N - 1:
                if weight > 0:
                    overlaps[i, i] = 1- weight
                    if rot:
                        overlaps[i, 0] = weight
                elif weight < 0:
                    weight = torch.abs(weight)
                    overlaps[i, i] = 1 - weight
                    overlaps[i, i - 1] = weight    
            elif 0 < i and i < N - 1:
                if weight > 0:
                    overlaps[i, i] = 1 - weight
                    overlaps[i, i + 1] = weight
                elif weight < 0:
                    weight = torch.abs(weight)
                    overlaps[i, i] = 1 - weight
                    overlaps[i, i - 1] = weight
        return overlaps

    def forward(self, out_dict, data_dict):
        rot_loss = 0
        tra_loss = 0
        qry_feats = out_dict['qry_pose_descs'] # (B, N, D)
        pos_feats = out_dict['pos_pose_descs'] # (B, N, D) 
        yaws = data_dict['relative_euler'][...,2:]
        B, N, D = qry_feats.shape
        shifts = yaws / (360 / self.theta)

        if 'neg_pos_pose_descs' in out_dict.keys() and 'pos_neg_pose_descs' in out_dict.keys():
            neg_pos_feats = out_dict['neg_pos_pose_descs'] # (B, N, D)
            pos_neg_feats = out_dict['pos_neg_pose_descs'] # (B, N, D)
            factor_t = torch.abs(out_dict['dist_t'])
            with_neg_margin = True
        else:
            with_neg_margin = False
        
        for idb in range(B):
            step = shifts[idb].item()
            if step >= 0:
                int_step = int(step)
                alpha = step - int_step
                qry_feats_s = (1 - alpha) * qry_feats[idb, ...].roll(int_step, 0) + alpha * qry_feats[idb, ...].roll(int_step + 1, 0)
                if with_neg_margin:
                    qry_feats_n = (1 - alpha) * neg_pos_feats[idb, ...].roll(int_step, 0) + alpha * neg_pos_feats[idb, ...].roll(int_step + 1, 0)
                    if self.trans_weighting:
                        x_disturbate_weight = (1 - alpha) * self.x_disturbate_weight.roll(int_step, 0) + alpha * self.x_disturbate_weight.roll(int_step + 1, 0)
                        y_disturbate_weight = (1 - alpha) * self.y_disturbate_weight.roll(int_step, 0) + alpha * self.y_disturbate_weight.roll(int_step + 1, 0)
            else:
                step = abs(step)
                int_step = int(step)
                alpha = step - int_step
                qry_feats_s = (1 - alpha) * qry_feats[idb, ...].roll(-int_step, 0) + alpha * qry_feats[idb, ...].roll(-int_step - 1, 0)
                if with_neg_margin:
                    qry_feats_n = (1 - alpha) * neg_pos_feats[idb, ...].roll(-int_step, 0) + alpha * neg_pos_feats[idb, ...].roll(-int_step - 1, 0)
                    if self.trans_weighting:
                        x_disturbate_weight = (1 - alpha) * self.x_disturbate_weight.roll(-int_step, 0) + alpha * self.x_disturbate_weight.roll(-int_step - 1, 0)
                        y_disturbate_weight = (1 - alpha) * self.y_disturbate_weight.roll(-int_step, 0) + alpha * self.y_disturbate_weight.roll(-int_step - 1, 0)

            feat_dists = torch.sqrt(pairwise_distance(qry_feats_s, pos_feats[idb], normalized=True))
            overlaps = torch.zeros(N, N).fill_diagonal_(1).cuda()
            pos_masks = torch.gt(overlaps, 0.1)
            neg_masks = torch.eq(overlaps, 0)
            pos_scales = torch.sqrt(overlaps * pos_masks.float())
            loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)
            if not torch.isnan(loss):
                rot_loss += loss

            if with_neg_margin:
                qry_pos_dis = torch.diag(feat_dists)
                neg_pos_dis = torch.diag(torch.sqrt(pairwise_distance(qry_feats_n, pos_neg_feats[idb], normalized=True)))
                if self.trans_weighting:
                    diff_x = torch.clamp((qry_pos_dis * x_disturbate_weight.cuda()).mean() - (neg_pos_dis * x_disturbate_weight.cuda()).mean() + factor_t[idb, 0] * self.d_weight, min=0.0)
                    diff_y = torch.clamp((qry_pos_dis * y_disturbate_weight.cuda()).mean() - (neg_pos_dis * y_disturbate_weight.cuda()).mean() + factor_t[idb, 1] * self.d_weight, min=0.0)
                    tra_loss += (diff_x + diff_y) / 2
                else:
                    diff = torch.clamp(qry_pos_dis.mean() - neg_pos_dis.mean() + self.d_factor, min=0.0)
                    tra_loss += diff
        
        rot_loss = rot_loss / B 
        tra_loss = tra_loss * self.tra_loss_weight / B 
        return {'rot_loss': rot_loss * self.loss_weight, 'tra_loss': tra_loss * self.loss_weight}

class LossFunction_LCD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.margin = cfg.loss.margin_triplet
        self.margin_pos = cfg.loss.margin_pos
        self.margin_neg = cfg.loss.margin_neg
        self.loss_triplet = nn.TripletMarginLoss(margin=self.margin, p=2, eps=1e-7, reduction='sum')
        self.loss_weight = cfg.loss.desc_loss_weight
        self.l_factor_a = 0.01
        self.l_factor_b = 0.04
    
    def hinge_loss(self, D_anchor, D_pos, D_neg, loop_dis_pos=None, loop_dis_neg=None):
        dis_pos = torch.norm(D_anchor - D_pos, p=2, dim=1) 
        if loop_dis_pos is not None:
            loss_pos = torch.clamp(dis_pos - loop_dis_pos, min=0.0)
        else:
            loss_pos = torch.clamp(dis_pos - self.margin_pos, min=0.0)
        dis_neg = torch.norm(D_anchor - D_neg, p=2, dim=1)
        dis_neg_tran = torch.norm(D_pos - D_neg, p=2, dim=1)
        if loop_dis_neg is not None:
            loss_neg = torch.clamp(loop_dis_neg - dis_neg, min=0.0)
            loss_neg_tran = torch.clamp(loop_dis_neg - dis_neg_tran, min=0.0)
        else:
            loss_neg = torch.clamp(self.margin_neg - dis_neg, min=0.0)
            loss_neg_tran = torch.clamp(self.margin_neg - dis_neg_tran, min=0.0)
        return torch.sum(loss_pos) + torch.sum(loss_neg) + torch.sum(loss_neg_tran) 

    def forward(self, out_dict, data_dict):
        anchor, positive, negatives = out_dict['qry_loop_desc'], out_dict['pos_loop_desc'], out_dict['neg_loop_desc']
        t = out_dict['rel_translation']
        dis = torch.sqrt(torch.pow(t[:, 0], 2) + torch.pow(t[:, 1], 2))
        loop_dis_pos = self.l_factor_a + self.l_factor_b * dis
        
        # print(anchor.shape, positive.shape, negatives[0].shape)
        # calculate triplet loss
        trip_loss = 0
        for n in range(len(negatives)):
            trip_loss += self.loss_triplet(anchor, positive, negatives[n])

        # calculate desc loss
        desc_loss = 0
        for n in range(len(negatives)):
            desc_loss += self.hinge_loss(anchor, positive, negatives[n], loop_dis_pos)
        
        return {'trip_loss': trip_loss * self.loss_weight, 'desc_loss': desc_loss * self.loss_weight}

class LossFunction_Total(nn.Module):
     def __init__(self, cfg):
        super().__init__()
        self.lcd_loss_func = LossFunction_LCD(cfg)
        self.with_pose_loss = cfg.model.with_local_feats
        self.pose_estimate = cfg.localizer.pose_estimate
        if self.with_pose_loss:
            if self.pose_estimate == 'regress' or self.pose_estimate == 'query':
                self.pose_loss_func = LossFunction_REG(cfg)
            elif self.pose_estimate == 'match':
                self.pose_loss_func = LossFunction_MTC(cfg)
            elif self.pose_estimate == 'polar':
                self.pose_loss_func = LossFunction_ROT(cfg)

     def forward(self, out_dict, data_dict):
        loss_dict = {}

        loop_loss_dict = self.lcd_loss_func(out_dict, data_dict)
        total_loss = loop_loss_dict['trip_loss'] + loop_loss_dict['desc_loss']
        loss_dict['trip_loss'] = loop_loss_dict['trip_loss']
        loss_dict['desc_loss'] = loop_loss_dict['desc_loss']
        if self.with_pose_loss:
            pose_loss_dict = self.pose_loss_func(out_dict, data_dict)
            if self.pose_estimate == 'polar':
                total_loss += pose_loss_dict['rot_loss']
                loss_dict['rot_loss'] = pose_loss_dict['rot_loss']
                total_loss += pose_loss_dict['tra_loss']
                loss_dict['tra_loss'] = pose_loss_dict['tra_loss']
            else:
                total_loss += pose_loss_dict['pose_loss']
                loss_dict['pose_loss'] = pose_loss_dict['pose_loss']
                
        loss_dict['loss'] = total_loss
        return loss_dict

def normalize_array(array, eps=1e-10):
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    normalized_array = array / (norm + eps)
    
    return normalized_array

@torch.no_grad()
def compute_metrics(out_dict, data_dict, margin=0.1, loop_thresh=0.1):
    metric_dict = {}
    # online compute loop closure metrics
    anchor, positive, negatives = out_dict['qry_loop_desc'], out_dict['pos_loop_desc'], out_dict['neg_loop_desc']
    B, D = anchor.shape
    acc_ratio, anchor_neg_ratio = 0, 0
    anchor_pos_dis  = torch.norm(anchor - positive, dim=1)
    for idn in range(len(negatives)):
        anchor_neg_dis =  torch.norm(anchor - negatives[idn], dim=1)
        acc_ratio += ((anchor_pos_dis - anchor_neg_dis) < margin).int().sum().item()
        anchor_neg_ratio += torch.where(anchor_neg_dis > loop_thresh)[0].shape[0]
    
    anchor_pos_ratio = torch.where(anchor_pos_dis < loop_thresh)[0].shape[0] / B
    anchor_neg_ratio = anchor_neg_ratio / (B * len(negatives))
    acc_ratio = acc_ratio / (B * len(negatives))

    # get hard samples
    idxs = data_dict['rdn_idx'] #(B, 3)
    # false_pos_indices = torch.where(anchor_pos_dis < loop_thresh)
    # false_neg_indices = torch.where(anchor_neg_dis > loop_thresh)
    # false_mgn_indices = torch.where((anchor_pos_dis - anchor_neg_dis) < margin)
    # hard_indices = torch.where((anchor_pos_dis < loop_thresh) | (anchor_neg_dis > loop_thresh) | ((anchor_pos_dis - anchor_neg_dis) < margin))
    # hard_indices = torch.where((anchor_neg_dis > loop_thresh) | ((anchor_pos_dis - anchor_neg_dis) < margin))
    hard_indices = torch.where((anchor_pos_dis - anchor_neg_dis) < margin)
    hard_idxs = idxs[hard_indices].detach().cpu().numpy() #(N, 3)

    metric_dict['pos_ratio'] = anchor_pos_ratio
    metric_dict['neg_ratio'] = anchor_neg_ratio
    metric_dict['acc'] = acc_ratio

    # online compute pose metrics
    if 'rel_q' in out_dict.keys() and 'rel_t' in out_dict.keys(): # for 6-dof fregression
        predict_t = tensor_to_array(out_dict['rel_t'])
        predict_q = tensor_to_array(out_dict['rel_q'])
        groundtruth_t = tensor_to_array(data_dict['relative_translation'])
        groundtruth_q = tensor_to_array(data_dict['relative_quaternion'])
        groundtruth_R = tensor_to_array(data_dict['relative_rotation'])

        rotation_error, translation_error = .0, .0
        B = predict_t.shape[0]
        for b in range(B):
            predict_R = quaternion_to_rotation(predict_q[b])
            T_pred = np.identity(4, dtype=np.float32)
            T_pred[0:3, 0:3] = predict_R
            T_pred[0:3, 3] = predict_t[b]

            T_groundtruth = np.identity(4, dtype=np.float32)
            T_groundtruth[0:3, 0:3] = groundtruth_R[b]
            T_groundtruth[0:3, 3] = groundtruth_t[b]

            RRE, RTE = compute_registration_error(T_pred, T_groundtruth)
            rotation_error += RRE
            translation_error += RTE
        
        metric_dict['mRRE'] = rotation_error / B
        metric_dict['mRTE'] = translation_error / B

    elif 'estimated_transforms' in out_dict.keys(): # for matching
        predict_T = tensor_to_array(out_dict['estimated_transforms'])
        groundtruth_t = tensor_to_array(data_dict['relative_translation'])
        groundtruth_q = tensor_to_array(data_dict['relative_quaternion'])
        groundtruth_R = tensor_to_array(data_dict['relative_rotation'])

        rotation_error, translation_error = .0, .0
        B = predict_T.shape[0]       

        for b in range(B):
            T_pred = predict_T[b, ...]
            T_groundtruth = np.identity(4, dtype=np.float32)
            T_groundtruth[0:3, 0:3] = groundtruth_R[b]
            T_groundtruth[0:3, 3] = groundtruth_t[b]

            RRE, RTE = compute_registration_error(T_pred, T_groundtruth)
            rotation_error += RRE
            translation_error += RTE

        metric_dict['mRRE'] = rotation_error / B
        metric_dict['mRTE'] = translation_error / B
    
    elif 'rel_yaw' in out_dict.keys() and 'rel_t' in out_dict.keys(): # for 3-dof fregression
        predict_t = tensor_to_array(out_dict['rel_t'])
        predict_q = tensor_to_array(out_dict['rel_yaw'])
        groundtruth_t = tensor_to_array(data_dict['relative_translation'])[..., :2]
        groundtruth_q = tensor_to_array(data_dict['relative_euler'])[..., 2:]

        rot_diff = predict_q - groundtruth_q
        rotation_error = np.linalg.norm(np.minimum(360 - abs(rot_diff), abs(rot_diff)), axis=1).mean()
        translation_error = np.linalg.norm(predict_t - groundtruth_t, axis=1).mean()
        # translation_error = np.linalg.norm(normalize_array(predict_t)  - normalize_array(groundtruth_t), axis=1).mean()
        
        metric_dict['mRRE'] = rotation_error
        metric_dict['mRTE'] = translation_error
    return metric_dict, hard_idxs