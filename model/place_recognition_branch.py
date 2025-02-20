import math
import sys
sys.path.append('../')
import numpy as np
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.voxel_utils.vox as vox
import utils.voxel_utils.basic as basic
import utils.voxel_utils.geom as geom
from utils.geometry import cam_to_pixel_torch, apply_transform_torch

from model.localizer import Bev_localizer
from model.bev_decoder import Bev_desc_decoder
from model.image_encoder import Image_encoder_res101, Image_encoder_res50
from model.attention import SpatialCrossAttention, VanillaSelfAttention

import pdb
import time

EPS = 1e-8

class MCVPR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.training = True
        self.dataset = cfg.data.dataset

        self.Z, self.Y, self.X = cfg.model.Z, cfg.model.Y, cfg.model.X
        self.feature_dim = cfg.model.feature_dim
        self.ffn_dim = cfg.model.ffn_dim
        self.output_dim = cfg.model.output_dim
        self.num_layers = cfg.model.num_layers
        self.desc_dim = cfg.model.desc_dim
        self.with_pos_emb = cfg.model.with_pos_emb

        self.use_normalize = cfg.model.normalize
        self.scene_centroid = cfg.model.scene_centroid

        bounds_x = cfg.model.xbounds
        bounds_y = cfg.model.ybounds
        bounds_z = cfg.model.zbounds
        self.bounds = [bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1], bounds_z[0], bounds_z[1]]

        # voxel utilities
        self.vox_util = vox.Vox_util(self.Z, self.Y, self.X,
                            scene_centroid=torch.from_numpy(np.array(self.scene_centroid)).float().cuda(),
                            bounds=self.bounds,
                            assert_cube=False)

        # image encoder
        if cfg.model.backbone == 'res101':
            self.img_encoder = Image_encoder_res101(self.feature_dim)
        elif cfg.model.backbone == 'res50':
            self.img_encoder = Image_encoder_res50(self.feature_dim)
        
        # BEV queries
        self.bev_queries = nn.Parameter(0.1*torch.randn(self.feature_dim, self.Z, self.X)) # C, X, Y
        if self.with_pos_emb:
            self.bev_queries_pos = nn.Parameter(0.1*torch.randn(self.feature_dim, self.Z, self.X)) # C, X, Y
        else:
            self.bev_queries_pos = None

        # attention layers
        self.self_attn_layers = nn.ModuleList([VanillaSelfAttention(dim=self.feature_dim, spatial_shapes=(self.Z, self.X)) for _ in range(self.num_layers)]) 
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(self.feature_dim) for _ in range(self.num_layers)])
        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(dim=self.feature_dim) for _ in range(self.num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(self.feature_dim) for _ in range(self.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.feature_dim, self.ffn_dim), 
            nn.ReLU(), 
            nn.Linear(self.ffn_dim, self.feature_dim)) for _ in range(self.num_layers)
        ])
        self.norm3_layers = nn.ModuleList([
            nn.LayerNorm(self.feature_dim) for _ in range(self.num_layers)
        ])       

        # global and local descriptor decoder
        self.with_local_feats = cfg.model.with_local_feats
        self.pose_estimate = cfg.localizer.pose_estimate
        self.norm_desc = not cfg.localizer.loop_interact
        self.bev_decoder = Bev_desc_decoder(cfg, with_normalize=self.norm_desc, training=self.training)

        # localizer for loop closure detection and pose estimation
        self.localizer = Bev_localizer(cfg, training=self.training)

        # freeze layers if finetuning
        if 'finetune' in cfg.train and cfg.train.finetune == True:
            self.freeze_encoder_parameters()
    
    def freeze_encoder_parameters(self):
        # for param in self.img_encoder.parameters():
        #     param.requires_grad = False
        # self.bev_queries.requires_grad = False
        # if self.bev_queries_pos is not None:
        #     self.bev_queries_pos.requires_grad = False
        # for param in self.self_attn_layers.parameters():
        #     param.requires_grad = False
        # for param in self.norm1_layers.parameters():
        #     param.requires_grad = False
        # for param in self.cross_attn_layers.parameters():
        #     param.requires_grad = Falses
        # for param in self.norm2_layers.parameters():
        #     param.requires_grad = False
        # for param in self.ffn_layers.parameters():
        #     param.requires_grad = False
        # for param in self.norm3_layers.parameters():
        #     param.requires_grad = False
        return

    def forward(self, data_dict, localizing=False, loop_detect=False):
        if localizing:
            return self.localizer(data_dict, training=False, loop_detect=loop_detect)
        
        out_dict = {}
        if self.training:
            # get query and positive descriptors
            qry_data_dict = {'query_img':  data_dict['query_imgs'], 'query_P':  data_dict['query_P'], 'query_K':  data_dict['query_K']}
            pos_data_dict = {'query_img':  data_dict['positive_imgs'], 'query_P':  data_dict['positive_P'], 'query_K':  data_dict['positive_K']}
            qry_desc_dict = self._forward(qry_data_dict)
            pos_desc_dict = self._forward(pos_data_dict)
            qry_loop_desc = qry_desc_dict['loop_desc']
            pos_loop_desc = pos_desc_dict['loop_desc']

            # get negatives descriptors
            neg_num = data_dict['negative_imgs'].shape[1]
            neg_loop_descs, neg_pose_descs = [], []
            for n in range(neg_num):
                neg_data_dict = {'query_img':  data_dict['negative_imgs'][:,n,...], 'query_P':  data_dict['negative_P'][:,n,...], 'query_K':  data_dict['negative_K'][:,n,...]}
                neg_desc_dict = self._forward(neg_data_dict)
                neg_loop_descs.append(neg_desc_dict['loop_desc'])
                if self.with_local_feats:
                    neg_pose_descs.append(neg_desc_dict['pose_desc'])

            out_dict['qry_loop_desc'] = qry_loop_desc
            out_dict['pos_loop_desc'] = pos_loop_desc
            out_dict['neg_loop_desc'] = neg_loop_descs

            # get ambiguous descriptors
            if 'ambiguous_imgs' in data_dict.keys():
                amb_data_dict = {'query_img': data_dict['ambiguous_imgs'], 'query_P': data_dict['ambiguous_P'], 'query_K': data_dict['query_K']}
                amb_desc_dict = self._forward(amb_data_dict)
                out_dict['amb_loop_desc'] = amb_desc_dict['loop_desc']

            # get pose descriptors
            if self.with_local_feats:
                qry_pose_desc = qry_desc_dict['pose_desc']
                pos_pose_desc = pos_desc_dict['pose_desc']
                if self.pose_estimate == 'match':
                    # the ego-points in each sample should be same exactly
                    out_dict['qry_ego_points'] =  qry_desc_dict['ego_points']
                    out_dict['pos_ego_points'] =  pos_desc_dict['ego_points']
                    out_dict['neg_ego_points'] =  neg_desc_dict['ego_points']
                    # to find correspondences online
                    out_dict['rel_R'] = data_dict['relative_rotation']
                    out_dict['rel_q'] = data_dict['relative_quaternion']
                    out_dict['rel_t'] = data_dict['relative_translation']
                    # bev masks
                    # out_dict['qry_bev_masks'] = qry_desc_dict['bev_mask']
                    # out_dict['pos_bev_masks'] = pos_desc_dict['bev_mask']
                    out_dict['bev_mask'] =  qry_desc_dict['bev_mask']
                elif self.pose_estimate == 'polar':
                    out_dict['rel_yaw'] = data_dict['relative_euler'][:, 2:]
                    out_dict['rel_translation'] = data_dict['relative_translation'][:, :2]
                out_dict['qry_pose_desc'] = qry_pose_desc
                out_dict['pos_pose_desc'] = pos_pose_desc
                out_dict['neg_pose_desc'] = neg_pose_descs

            # localizing
            out_dict = self.localizer(out_dict)
            return out_dict
        else:
            # directly return loop descriptor and pose descriptor of single frame data
            return self._forward(data_dict)

    def _forward(self, data_dict):
        # data preparation
        query_img = data_dict['query_img'] #(b, 4, c, h, w)
        pix_T_cams_qry = data_dict['query_K'] # (B, 4, 3, 3)
        camXs_T_cam0_qry = data_dict['query_P'] # (B, 4, 4, 4)
        if len(query_img.shape) == 4:
            query_img = query_img.unsqueeze(0)
            pix_T_cams_qry = pix_T_cams_qry.unsqueeze(0)
            camXs_T_cam0_qry = camXs_T_cam0_qry.unsqueeze(0)

        B, S, C, H, W = query_img.shape 
        B0 = B * S
        device = query_img.device

        __p = lambda x: basic.pack_seqdim(x, B)
        __u = lambda x: basic.unpack_seqdim(x, B)

        query_img = __p(query_img) #(b *4, c, h, w)
        pix_T_cams_qry = __p(pix_T_cams_qry)  #(b * 4, 3, 3)
        camXs_T_cam0_qry = __p(camXs_T_cam0_qry) #(b * 4, 4, 4)

        # t1 = time.time()
        query_feats = self.img_encoder(query_img)
        # t2 = time.time()
        # print('image time', t2 - t1)
        _, C, Hf, Wf = query_feats.shape
        query_feats = __u(query_feats) #(b, 4, C,hf, wf)

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # compute the image locations (no flipping for now)
        # xyz_mem_ = basic.gridcloud3d(B0, Z, Y, X, norm=False, device=device) # (B0, Z*Y*X, 3）
        # xyz_cam0_ = self.vox_util.Mem2Ref(xyz_mem_, Z, Y, X, assert_cube=False) #(B0, Z*Y*X, 3)
        xyz_mem_ = basic.gridcloud3d(B0, Y, X, Z, norm=False, device=device) # (B0, Z*Y*X, 3）
        xyz_cam0_ = self.vox_util.Mem2Ref(xyz_mem_, Y, X, Z, assert_cube=False) #(B0, Z*Y*X, 3)
        xyz_cam0_[:,:,2] =  -xyz_cam0_[:,:,2]

        # project query points
        xyz_camXs_ = apply_transform_torch(xyz_cam0_, camXs_T_cam0_qry)
        xy_camXs_ = cam_to_pixel_torch(xyz_camXs_, pix_T_cams_qry)
        xy_camXs = __u(xy_camXs_) #(b, s, X*Y*Z, 2)

        # normalize and get masks
        reference_points_cam = xy_camXs_.reshape(B, S, Y, Z, X, 3).permute(1, 0, 4, 3, 2, 5).flatten(2, 3)
        reference_points_cam[..., 0:1] = reference_points_cam[..., 0:1] / float(W) #(s, b, X*Y, Z, 2)
        reference_points_cam[..., 1:2] = reference_points_cam[..., 1:2] / float(H)  #(s, b, X*Y, Z, 2)
        bev_mask = ((reference_points_cam[..., 1] > 0.0)
                    & (reference_points_cam[..., 1] < 1.0)
                    & (reference_points_cam[..., 0] < 1.0)
                    & (reference_points_cam[..., 0] > 0.0)
                    & (reference_points_cam[..., 2] > 0.0)).squeeze(-1) #(s, b, Z*X, Y)
        # save bev_mask and reference points for checking
        # np.save('./reference_points_ego.npy', xyz_cam0_.reshape(4, 20, 100, 100, 3).permute(0, 3, 2, 1, 4).detach().cpu().numpy())
        # np.save('./reference_points_cam.npy', reference_points_cam.reshape(4, 100, 100, 20, 3).detach().cpu().numpy())
        # np.save('./bev_mask.npy', bev_mask[:,0,:,:].reshape(4, 100, 100, 20).detach().cpu().numpy())
        # reference_points_ego = xyz_cam0_.reshape(B, S, Y, Z, X, 3).permute(1, 0, 4, 3, 2, 5).flatten(2, 3) # checking 
        
        reference_points_cam = reference_points_cam[..., 0:2]
        # np.save('./ref_points.npy', reference_points_cam.reshape(4, 1, 100, 100, 20, 2).detach().cpu().numpy())
        # reference_points_ego = reference_points_ego[...,0:2] # checking
        bev_queries = self.bev_queries.clone().unsqueeze(0).repeat(B,1,1,1).reshape(B, self.feature_dim, -1).permute(0,2,1) # (b, X*Y, D)
        bev_queries_pos = self.bev_queries_pos.clone().unsqueeze(0).repeat(B,1,1,1).reshape(B, self.feature_dim, -1).permute(0,2,1) # B, X*Y, C
        bev_keys = query_feats.reshape(B, S, C, Hf*Wf).permute(1, 3, 0, 2) # (s, Hf*Wf, b, D)
        spatial_shapes = bev_queries.new_zeros([1, 2]).long() #(1,2)
        spatial_shapes[0, 0] = Hf
        spatial_shapes[0, 1] = Wf

        # t3 = time.time()
        # apply attention modules
        for i in range(self.num_layers):
            bev_queries = self.self_attn_layers[i](bev_queries, bev_queries_pos)
            bev_queries = self.norm1_layers[i](bev_queries)
            bev_queries = self.cross_attn_layers[i](bev_queries, bev_keys, bev_keys, 
                query_pos= bev_queries_pos,
                reference_points_cam = reference_points_cam,
                spatial_shapes = spatial_shapes, 
                bev_mask = bev_mask,
            )
            bev_queries = self.norm2_layers[i](bev_queries)
            bev_queries = bev_queries + self.ffn_layers[i](bev_queries)
            bev_queries = self.norm3_layers[i](bev_queries)
        feat_bev = bev_queries.permute(0, 2, 1).reshape(B, self.feature_dim, self.Z, self.X) # (b, D, X, Y)
        # t4 = time.time()
        # print('BEV', t4 - t3)
        out_dict = self.bev_decoder(feat_bev, ground_bev=None, ego_points=None)
        if self.pose_estimate == 'match':
            out_dict['bev_mask'] = bev_mask_ground
        return out_dict

def create_model(cfg):
    return MCVPR(cfg)

def test_model_with_sync_data():
    from configs.cfg_place_recognition_baseline import make_cfg
    cfg = make_cfg()

    model = MCVPR(cfg).to('cuda')
    model.training=False
    data_dict = {}
    data_dict['query_img'] = torch.from_numpy(np.random.rand(2, 4, 3, 240, 427)).float().to('cuda')

    temp_K = np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]])
    temp_T = np.eye(4)

    data_dict['query_K'] = torch.from_numpy(temp_K).float().to('cuda').repeat(2, 4, 1, 1)
    data_dict['query_P'] = torch.from_numpy(temp_T).float().to('cuda').repeat(2, 4, 1, 1)

    out = model(data_dict)
    pdb.set_trace()
    return

def test_model_with_real_data():
    from configs.cfg_place_recognition_baseline import make_cfg
    from dataset.collate import SimpleSingleCollateFnPackMode
    from dataset.nio_visual_loop_dataset import NioVisualLoopDataset
    from torch.utils.data import Dataset, DataLoader

    dataset = NioVisualLoopDataset(dataset_path='/map-hl/gary.huang/Visual_Loop_Closure/training', loop_closure_path='../dataset/loop_closure.pkl')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=SimpleSingleCollateFnPackMode(), num_workers=4)
    for iteration, data_dict in enumerate(dataloader):
        pdb.set_trace()

    return


if __name__ == '__main__':
    test_model_with_sync_data()
    # test_model_with_real_data()