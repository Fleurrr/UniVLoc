import argparse

import os
import cv2
import json
import time
import copy
import torch
import struct
import colorsys
import numpy as np
import open3d as o3d

from model.place_recognition_branch import create_model
from utils.torch_utils import to_cuda, all_reduce_tensors, release_cuda, initialize

from tqdm import tqdm
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
from utils.registration import compute_registration_error
from utils.fisheye import load_fisheye_json, fast_fisheye_distortion
from utils.io_tools import read_json_from_file, read_csv,  read_txt, make_dir, save_dict, load_pkl, read_json_from_file, load_pkl
from utils.geometry import quaternion_to_rotation, rotation_to_euler, get_transform_from_rotation_translation, inverse_transform, get_rotation_translation_from_transform, apply_transform, euler_to_rotation

import pdb
import time
import pickle

initialize(
        seed=8888,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
)

def save_loop_imgs(imgs, name='qry'):
    imgs = imgs[0].detach().cpu().numpy()
    for ids in range(4):
        cv2.imwrite('cam_' + str(ids) + '_' + name + '.png', imgs[ids].transpose(1,2,0))
    return

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='../output/place_recognition_baseline_polar/snapshots/snapshot.pth.tar', help="model weights file")
    parser.add_argument('--mode', choices=['indoor', 'outdoor'], default='indoor')
    parser.add_argument('--with_prior', default=False, action='store_true', help='search with prior')
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    from configs.cfg_joint_baseline import make_cfg
    cfg = make_cfg()

    model = create_model(cfg).cuda()
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()
    model.training = False

    from dataset.collate import SimpleSingleCollateFnPackMode
    from dataset.nio_visual_loop_dataset import NioVisualLoopDataset

    if args.mode == 'indoor':
        pkl_path = './dataset/loop_closure_eval_indoor_vis_2m.pkl'
    elif args.mode == 'outdoor':
        pkl_path = './dataset/loop_closure_eval_outdoor_vis_2m.pkl'
    
    cfg = make_cfg()
    dataset = NioVisualLoopDataset(cfg.data.dataset_path, cfg.data.with_amb, \
                                                                                                             loop_closure_path=pkl_path, \
                                                                                                             img_width=cfg.data.img_width, img_height=cfg.data.img_height,  \
                                                                                                             dataset_length=cfg.data.dataset_length, neg_num=cfg.data.neg_num, \
                                                                                                             mode='train', demo=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=SimpleSingleCollateFnPackMode(), num_workers=1)
    mrye, mrte, filtered, loop_filtered, loop_unfiltered = 0, 0, 0, 0, 0
    info = []
    save = False
    with_neg = False
    loop_iter = 3

    for _ in range(loop_iter):
        for iteration, data_dict in tqdm(enumerate(dataloader)):
            # for query data
            qry_data_dict = {'query_img':  data_dict['query_imgs'].float(), 'query_P':  data_dict['query_P'].float(), 'query_K':  data_dict['query_K'].float()}
            qry_out_dict = release_cuda(model(to_cuda(qry_data_dict)))
            qry_bev = qry_out_dict ['bev'].squeeze()
            qry_desc = qry_out_dict['loop_desc'].squeeze()
            if save:
                np.save('./output/qry_imgs.npy', data_dict['query_imgs'].detach().cpu().numpy())
                # np.save('./output/qry_bev.npy', qry_bev)
                # np.save('./output/qry_loop_desc.npy', qry_desc)

            # for positive data
            pos_data_dict = {'query_img':  data_dict['positive_imgs'], 'query_P':  data_dict['positive_P'], 'query_K':  data_dict['positive_K']}
            pos_out_dict = release_cuda(model(to_cuda(pos_data_dict)))
            pos_bev = pos_out_dict ['bev'].squeeze()
            pos_desc = pos_out_dict['loop_desc'].squeeze()
            if save:
                np.save('./output/pos_imgs.npy', data_dict['positive_imgs'].detach().cpu().numpy())
                # np.save('./output/pos_bev.npy', pos_bev)
                # np.save('./output/pos_loop_desc.npy', pos_desc)

            # for negative data
            if with_neg:
                neg_data_dict = {'query_img':  data_dict['negative_imgs'][0], 'query_P':  data_dict['negative_P'][0], 'query_K':  data_dict['negative_K'][0]}
                neg_out_dict = release_cuda(model(to_cuda(neg_data_dict)))
                neg_bev = neg_out_dict ['bev'].squeeze()
                neg_desc = neg_out_dict['loop_desc'].squeeze()

            # for global-featur-based loop closure detection
            t1 = time.time()
            dist = np.linalg.norm(qry_desc - pos_desc)
            t2 = time.time()
            print('TL', t2 - t1)
            if with_neg:
                dist_neg = np.linalg.norm(qry_desc - neg_desc)
            if dist > 0.05:
                loop_filtered += 1
            if with_neg:
                if dist_neg < 0.05:
                    loop_unfiltered += 1

            # for positive pose estimation
            if save:
                np.save('./output/euler.npy', data_dict['relative_euler'].detach().cpu().numpy())
                np.save('./output/translation.npy', data_dict['relative_translation'].detach().cpu().numpy())
            gt_yaw = data_dict['relative_euler'][0][2].numpy()
            gt_t = data_dict['relative_translation'][0,:2].numpy()
            pose_input_dict = {'pose_desc_1': torch.from_numpy(qry_bev).float().unsqueeze(0), 'pose_desc_2': torch.from_numpy(pos_bev).float().unsqueeze(0)}
            # pose_input_dict['rel_t'] = data_dict['relative_translation'][:, :2]
            if args.with_prior:
                if gt_yaw >= 15:
                    pose_input_dict['prior_yaw'] = gt_yaw - 15
                else:
                    pose_input_dict['prior_yaw'] = gt_yaw + 15
            t3 = time.time()
            pose_out_dict = release_cuda(model(to_cuda(pose_input_dict), localizing=True))
            t4 = time.time()
            print('ML', t4 - t3)
            rel_yaw = pose_out_dict['rel_yaw']
            rel_t = pose_out_dict['rel_t'][0][0]
            cost = pose_out_dict['min_costs']

            # for negative pose estimation
            if with_neg:
                neg_pose_input_dict = {'pose_desc_1': torch.from_numpy(qry_bev).float().unsqueeze(0), 'pose_desc_2': torch.from_numpy(neg_bev).float().unsqueeze(0)}
                neg_pose_out_dict = release_cuda(model(to_cuda(neg_pose_input_dict), localizing=True))
                neg_cost = neg_pose_out_dict['min_costs']

            if with_neg:
                print(dist, dist_neg, cost, neg_cost)
                # if dist_neg < 0.05:
                #     pdb.set_trace()

            # if save:
            #     qry_translated_bev = pose_out_dict['qry_translated_bev'].squeeze()
            #     qry_pose_desc = pose_out_dict['qry_polar_bev'].squeeze()
            #     pos_pose_desc = pose_out_dict['pos_polar_bev'].squeeze()
            #     rotated_bev = pose_out_dict['rotated_bev'].squeeze()
            #     np.save('./output/qry_translated_bev.npy', qry_translated_bev)
            #     np.save('./output/qry_pose_desc.npy', qry_pose_desc)
            #     np.save('./output/pos_pose_desc.npy', pos_pose_desc)
            #     np.save('./output/rotated_bev.npy', rotated_bev) 

            # yaw and translation estimation
            diff = abs(rel_yaw - gt_yaw)
            rye = min(360 - diff, diff)
            print("estimated/gt yaw:", rel_yaw, gt_yaw)
            rte = np.linalg.norm(gt_t - rel_t)
            if cost > 0.745:
                filtered += 1
            else:
                mrte += rte
                mrye += rye
            
            print("estimated/gt translation:", rel_t, gt_t)
            print("optimal cost:", cost)
            print("loop distance", dist)
            print('rye/mrye/rte/mrte/filtered/loop_filtered:', rye, mrye / (iteration + 1 - filtered + 0.0001), rte, mrte / (iteration + 1 - filtered + 0.0001), filtered)

            if cfg.localizer.searching:
                if with_neg:
                    info.append([rel_yaw, gt_yaw, rye, cost, rel_t[0], rel_t[1], gt_t[0], gt_t[1], rte, dist, dist_neg, cost, neg_cost])
                else:
                    info.append([rel_yaw, gt_yaw, rye, cost, rel_t[0], rel_t[1], gt_t[0], gt_t[1], rte])
            else:
                info.append([rel_yaw, gt_yaw, rye, cost, dist])
        
            # search_result = cv2.imread('search.png')
            # search_result[round((rel_t[0] + 2) / 0.2), round((rel_t[1] + 2) / 0.2)] = [0, 0, 255] 
            # search_result[round((gt_t[0] + 2) / 0.2), round((gt_t[1] + 2 )/ 0.2)] = [255, 0, 0] 
            # cv2.imwrite('pointed_search.png', search_result)
            # pdb.set_trace()
            if save:
                pdb.set_trace()

    if args.mode == 'outdoor':
        if args.with_prior:
            with open('no_pts_outdoor_search_translation_pose_info_with_prior.pkl', 'wb') as file:
                pickle.dump(info, file)
        else:
            with open('no_pts_outdoor_search_translation_pose_info.pkl', 'wb') as file:
                pickle.dump(info, file)            
    elif args.mode == 'indoor':
        if args.with_prior:
            with open('no_pts_indoor_search_translation_pose_info_with_prior.pkl', 'wb') as file:
                pickle.dump(info, file)
        else:
             with open('no_pts_indoor_search_translation_pose_info.pkl', 'wb') as file:
                pickle.dump(info, file)           
    return

if __name__ == "__main__":
    main()