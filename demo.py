import argparse

import os
import sys
import cv2
import math
import json
import time
import copy
import torch
import struct
import random
import colorsys
import traceback
import numpy as np
import open3d as o3d

from model.place_recognition_branch import create_model
from dataset.collate import SimpleSingleCollateFnPackMode
from dataset.nio_visual_loop_demo_dataset import NioVisualLoopDemoDataset
from utils.torch_utils import to_cuda, all_reduce_tensors, release_cuda, initialize

from tqdm import tqdm
# from loguru import logger
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
        seed=8888, #1234
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dir", default='./', help="path of the raw session")
    parser.add_argument("--input_json", default='./online_infer/cloud_filter.json', help="filtered cloud json file")
    parser.add_argument("--result_dir", default='./online_infer/', help="model output path")
    parser.add_argument("--output_style", default=['1'], type=list) # stage 1 output; stage 2 output
    parser.add_argument("--with_yaw", default=False, action='store_true', help='give yaw estimation')
    parser.add_argument("--debug", default=False, action='store_true', help='debug mode')
    parser.add_argument("--model_path", default='../output/place_recognition_baseline_polar/snapshots/snapshot.pth.tar', help="model weights file")
    return parser

def scale_intrinsics(K, sx, sy):
    pose_aug = np.eye(3)
    pose_aug[0, 0] = sx
    pose_aug[1, 1] = sy
    K = pose_aug @ K
    return K

def draw_pc_in_image(points_cam, img, EPS=1e-6):
    H, W = img.shape[0], img.shape[1]
    for i in range(points_cam.shape[0]):
        py, px, pz = int(points_cam[i][1] / (points_cam[i][2] + EPS)), int(points_cam[i][0] / (points_cam[i][2] + EPS)), int(points_cam[i][2])
        if py < 0 or py >= H or px < 0 or px >= W or pz < 0:
            continue
        else:
            color = get_color_from_z(pz)
            img[py, px, :] = color
    return img

def get_color_from_z(z):  
    z_normalized = z / 160.0  
    hue = z_normalized if z_normalized <= 0.6667 else 1 - (z_normalized - 0.6667)  
    saturation = 1.0
    value = 1.0
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)  
    rgb = tuple(int(c * 255) for c in rgb)  
    return np.array(rgb)[::-1]

def get_submap(base_pcd, session_path, base_path, input_json, submap_size=15, gap=4, crop=True, max_range=30):
    keyframe_id = int(base_path.split('/')[-1].split('_')[-1].replace('.bin', ''))
    keyframe_pose_key = base_path.split('/')[0] + '_pose_list'
    base_pose = np.array(input_json[keyframe_pose_key][base_path]['T_lidar'])
    for i in range(submap_size):
        src_keyframe_id = keyframe_id + i + 1
        src_keyframe_path = base_path.replace('key_frame_' + str(keyframe_id) + '.bin', 'key_frame_' + str(src_keyframe_id) + '.bin')
        if src_keyframe_path in input_json[keyframe_pose_key].keys():
            src_keyframe_pose = np.array(input_json[keyframe_pose_key][src_keyframe_path]['T_lidar'])
            src_pcd = load_pcd_in_bin(os.path.join(session_path, src_keyframe_path), o3d_format=True)
            src_pcd.transform(np.linalg.inv(base_pose) @ src_keyframe_pose)
            base_pcd = base_pcd + src_pcd
        else:
            continue

    for i in range(0, submap_size * gap, 2):
        src_keyframe_id = keyframe_id - i - 1
        if src_keyframe_id <= 0:
            continue
        src_keyframe_path = base_path.replace('key_frame_' + str(keyframe_id) + '.bin', 'key_frame_' + str(src_keyframe_id) + '.bin')
        if src_keyframe_path in input_json[keyframe_pose_key].keys():
            src_keyframe_pose = np.array(input_json[keyframe_pose_key][src_keyframe_path]['T_lidar'])
            src_pcd = load_pcd_in_bin(os.path.join(session_path, src_keyframe_path), o3d_format=True)
            src_pcd.transform(np.linalg.inv(base_pose) @ src_keyframe_pose)
            base_pcd = base_pcd + src_pcd
        else:
            continue

    if crop:
        points = np.asarray(base_pcd.points)
        x_min, x_max = -max_range, max_range
        y_min, y_max = -max_range, max_range

        mask = (points[:, 1] >= x_min) & (points[:, 1] <= x_max) & \
            (points[:, 2] >= y_min) & (points[:, 2] <= y_max)
        filtered_points = points[mask]

        base_pcd = o3d.geometry.PointCloud()
        base_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return base_pcd

def load_pcd_in_bin(bin_path, need_intensity=False, need_downsample=True, o3d_format=True):
    with open(bin_path, 'rb') as file:
        num_points = struct.unpack('I', file.read(4))[0]
        data = file.read(num_points * 7)
        dt = np.dtype([('x', '<i2'), ('y', '<i2'), ('z', '<i2'), ('w', 'u1')])
        point_data = np.frombuffer(data, dtype=dt)
        point_data = np.array(point_data.tolist())

        points = point_data[:, :3].astype(np.float32) * 1e-2
        colors = np.zeros((num_points, 3), dtype=np.float32)
        colors[:, 0] = point_data[:, 3] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if need_intensity:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if need_downsample:
            pcd = pcd.voxel_down_sample(0.1)
        if o3d_format:
            return pcd
        else:
            return np.asarray(pcd.points)

def get_pointcloud(session_dir, bin_path, input_json=None, o3d_format=False):
    path_info = bin_path.split('/')
    lidar_parameter_path = os.path.join(session_dir, path_info[0], path_info[1], 'parameters', 'lidar/lidar_front.json')
    lidar_param = read_json_from_file(lidar_parameter_path)
    lidar_Rotation, lidar_translation = lidar_param['extrinsic_param']['rotation'], lidar_param['extrinsic_param']['translation']
    lidar_Rotation = Rotation.from_rotvec(lidar_Rotation)
    lidar_trans_lid2veh_ = np.eye(4)
    lidar_trans_lid2veh_[:3, :3] = lidar_Rotation.as_matrix()
    lidar_trans_lid2veh_[:3, 3] = lidar_translation
    lidar_trans_veh2lid_ = np.linalg.inv(lidar_trans_lid2veh_)
    pcd_lid = load_pcd_in_bin(os.path.join(session_dir, bin_path), o3d_format=True)  
    if input_json is not None:
        pcd_lid = get_submap(pcd_lid, session_dir, bin_path, input_json)
    if o3d_format:
        return pcd_lid
    else:
        return np.asarray(pcd_lid.points)

def load_data(session_dir, bin_path, img_height=256, img_width=448, with_lidar=False,  input_json=None, img_norm=False):
    # gert necesaary data path from lidar path
    path_info = bin_path.split('/')
    image_path = os.path.join(session_dir, path_info[0], path_info[1], 'images')
    parameter_path = os.path.join(session_dir, path_info[0], path_info[1], 'parameters', 'camera')
    keyframe_name = path_info[-1].replace('.bin', '')

    fw_img_path = os.path.join(image_path, 'FW', keyframe_name + '.jpg')
    rn_img_path = os.path.join(image_path, 'RN', keyframe_name + '.jpg')
    svc_left_img_path = os.path.join(image_path, 'SVC-Left', keyframe_name + '.jpg')
    svc_right_img_path = os.path.join(image_path, 'SVC-Right', keyframe_name + '.jpg')

    if not os.path.exists(fw_img_path):
        return {}
    if not os.path.exists(rn_img_path):
        return {}
    if not os.path.exists(svc_left_img_path):
        return {}
    if not os.path.exists(svc_right_img_path):
        return {} 

    fw_camera_parameter_path = os.path.join(parameter_path, 'front_wide/front_wide.json')
    rn_camera_parameter_path = os.path.join(parameter_path, 'rear_narrow/rear_narrow.json')
    svc_left_camera_parameter_path = os.path.join(parameter_path, 'svc_left/svc_left.json')
    svc_right_camera_parameter_path = os.path.join(parameter_path, 'svc_right/svc_right.json')
    
    # load raw images
    fw_img = cv2.imread(fw_img_path)
    rn_img = cv2.imread(rn_img_path)
    fw_img = cv2.resize(fw_img, (img_width, img_height))
    rn_img = cv2.resize(rn_img, (img_width, img_height))

    # load pointcloud
    if with_lidar:
        lidar_parameter_path = os.path.join(session_dir, path_info[0], path_info[1], 'parameters', 'lidar/lidar_front.json')
        lidar_param = read_json_from_file(lidar_parameter_path)
        lidar_Rotation, lidar_translation = lidar_param['extrinsic_param']['rotation'], lidar_param['extrinsic_param']['translation']
        lidar_Rotation = Rotation.from_rotvec(lidar_Rotation)
        lidar_trans_lid2veh_ = np.eye(4)
        lidar_trans_lid2veh_[:3, :3] = lidar_Rotation.as_matrix()
        lidar_trans_lid2veh_[:3, 3] = lidar_translation
        lidar_trans_veh2lid_ = np.linalg.inv(lidar_trans_lid2veh_)
        pcd_lid = load_pcd_in_bin(os.path.join(session_dir, bin_path), o3d_format=True)
        if input_json is not None:
            pcd_lid = get_submap(pcd_lid, session_dir, bin_path, input_json)
        # change point cloud to ego-vehicle system
        pcd_lid = np.array(pcd_lid.points)
        pcd_veh = apply_transform(pcd_lid, lidar_trans_lid2veh_)

    # fw camera pre-processing
    fw_cam_param = read_json_from_file(fw_camera_parameter_path)
    fw_K, fw_Rotation, fw_translation =fw_cam_param['intrinsic_param']['camera_matrix'], fw_cam_param['extrinsic_param']['rotation'], fw_cam_param['extrinsic_param']['translation']
    fw_origin_h, fw_origin_w = fw_cam_param['intrinsic_param']['camera_height'], fw_cam_param['intrinsic_param']['camera_width']
    fw_Rotation = Rotation.from_rotvec(fw_Rotation)
    fw_trans_cam2veh_ = np.eye(4)
    fw_trans_cam2veh_[:3, :3] = fw_Rotation.as_matrix()
    fw_trans_cam2veh_[:3, 3] = fw_translation
    fw_trans_veh2cam_ = np.linalg.inv(fw_trans_cam2veh_)
    fw_K = scale_intrinsics(fw_K, img_width/fw_origin_w, img_height/fw_origin_h)

    # rn camera pre-processing
    rn_cam_param = read_json_from_file(rn_camera_parameter_path)
    rn_K, rn_Rotation, rn_translation =rn_cam_param['intrinsic_param']['camera_matrix'], rn_cam_param['extrinsic_param']['rotation'], rn_cam_param['extrinsic_param']['translation']
    rn_origin_h, rn_origin_w = fw_cam_param['intrinsic_param']['camera_height'], fw_cam_param['intrinsic_param']['camera_width']
    rn_Rotation = Rotation.from_rotvec(rn_Rotation)
    rn_trans_cam2veh_ = np.eye(4)
    rn_trans_cam2veh_[:3, :3] = rn_Rotation.as_matrix()
    rn_trans_cam2veh_[:3, 3] = rn_translation
    rn_trans_veh2cam_ = np.linalg.inv(rn_trans_cam2veh_)
    rn_K = scale_intrinsics(rn_K, img_width/rn_origin_w, img_height/rn_origin_h)

    # svc-left camera pre-processing
    svc_left_img = cv2.imread(svc_left_img_path)
    svc_left_cam_param = load_fisheye_json(svc_left_camera_parameter_path, 'SVC-Left')
    svc_left_img, svc_left_K = fast_fisheye_distortion(svc_left_img, svc_left_cam_param, img_width, img_height)
    svc_left_trans_veh2cam =  svc_left_cam_param['trans_veh2cam_']

    # svc-right camera pre-processing
    svc_right_img = cv2.imread(svc_right_img_path)
    svc_right_cam_param = load_fisheye_json(svc_right_camera_parameter_path, 'SVC-Right')
    svc_right_img, svc_right_K = fast_fisheye_distortion(svc_right_img, svc_right_cam_param, img_width, img_height)
    svc_right_trans_veh2cam = svc_right_cam_param['trans_veh2cam_']

    # stack camera parameters and imgs
    imgs = np.stack([fw_img, rn_img, svc_left_img, svc_right_img])
    P = np.stack([fw_trans_veh2cam_, rn_trans_veh2cam_, svc_left_trans_veh2cam, svc_right_trans_veh2cam])
    K = np.stack([fw_K, rn_K, svc_left_K, svc_right_K])

    # prepare data dict
    data_dict = {}
    data_dict['query_img'] = torch.from_numpy(np.transpose(imgs, [0,3,1,2])).unsqueeze(0).float()
    data_dict['query_P'] = torch.from_numpy(P).unsqueeze(0).float()
    data_dict['query_K'] = torch.from_numpy(K).unsqueeze(0).float()
    data_dict = to_cuda(data_dict)
    if with_lidar:
        data_dict['query_pcd'] = pcd_veh
    
    return data_dict

def sample_dict_items(dictionary, num_samples):
    if num_samples == -1:
        return dictionary
    keys = list(dictionary.keys()) 
    sampled_keys = random.sample(keys, min(num_samples, len(keys)))
    sampled_items = {key: dictionary[key] for key in sampled_keys} 
    return sampled_items

def sample_dict(data, N=15000, pre_filtered=False, loop_sample=2, keep_top_rate=3, thresh_dict=None, session_dict=None):
    # 1. randomlt filter out several keys
    if pre_filtered:
        data = sample_dict_items(data, len(data) // loop_sample)

    # 2. keep the part of the loop with higher loop_thresh
    if thresh_dict is not None:
        sampled_dict = {key: [] for key in data}
        for key in sampled_dict:
            query_loop_paths = data[key]
            query_loop_thresh = thresh_dict[key]
            query_loop_session = session_dict[key]

            grouped = {}  
            for path, thresh, session in zip(query_loop_paths, query_loop_thresh, query_loop_session):  
                if session not in grouped:  
                    grouped[session] = []  
                grouped[session].append((path, thresh))             
            for key_a in grouped:  
                grouped[key_a] = sorted(grouped[key_a], key=lambda x: x[1])              
            filtered_paths = []  
            for key_b, value in grouped.items():  
                num_to_keep = math.ceil(len(value) / keep_top_rate)
                filtered_paths.extend([path for path, _ in value[:num_to_keep]])
            sampled_dict[key] = filtered_paths

        data = sampled_dict

    # 3. if still too many, randomly filtered out loops
    if N != -1:
        sorted_keys = sorted(data.keys(), key=lambda k: len(data[k]))
        sampled_dict = {key: [] for key in data}
        total_elements = 0

        for key in sorted_keys:
            if len(data[key]) <= 2:
                sampled_dict[key] = data[key]
                total_elements += len(data[key])

        remaining_elements = N - total_elements
        if remaining_elements <= 0:
            sampled_dict = {k: v for k, v in sampled_dict.items() if v}
            return sampled_dict

        keys_to_sample = [key for key in sorted_keys if len(data[key]) > 2]
        key_index = 0

        while remaining_elements > 0 and key_index < len(keys_to_sample):
            key = keys_to_sample[key_index]
            max_elements_to_sample = min(len(data[key]), remaining_elements)
            sampled_indices = random.sample(range(len(data[key])), max_elements_to_sample)
            sampled_dict[key] = [data[key][i] for i in sampled_indices]

            remaining_elements -= len(sampled_dict[key])
            key_index += 1

        sampled_dict = {k: v for k, v in sampled_dict.items() if v}
    else:
        sampled_dict = data
    sampled_dict = dict(sorted(sampled_dict.items()))
    return sampled_dict

def main():
    parser = make_parser()
    args = parser.parse_args()
    with_lidar=False
    if args.with_yaw:
        from configs.cfg_joint_baseline import make_cfg
        import small_gicp
        with_lidar=True
        args.output_style = '2'
    else:
        from configs.cfg_place_recognition_baseline import make_cfg
    cfg = make_cfg()

    if not os.path.exists(args.input_json):
        return

    model = create_model(cfg).cuda()
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict["model"], strict=False)
    model.eval()
    model.training = False

    loop_thresh = cfg.infer.loop_thresh
    match_thresh = cfg.infer.match_thresh
    reg_thresh = cfg.infer.reg_thresh
    delta_z_thresh = cfg.infer.delta_z_thresh
    p_thresh = cfg.infer.p_thresh
    r_thresh = cfg.infer.r_thresh
    use_dual_reg = cfg.infer.dual_reg

    f = open(args.input_json)
    input_json = json.load(f)
    if  '2' in args.output_style:
        place_name = args.input_json.split('_')[-1].replace('.json', '')
        if args.debug:
            out_path = os.path.join(args.result_dir, 'cloud_matcher_' + place_name)
        else:
            out_path = args.result_dir
        if not os.path.exists(out_path):
            os.mkdir(out_path)
    if '1' in args.output_style:
        out_json = {}
        out_json['cloud_pair_list'] = {}
        loop_thresh_dict  = {}
        loop_session_dict = {}

    preserved_descs = {}
    preserved_bevs = {}
    idx = 0
    if with_lidar:
        preserved_lidars = {}

    # for map without loop
    if len(input_json['cloud_pair_list']) == 0:
        print('No valid loops!')
        return 
    
    # pre-generate descriptors
    idy = 0
    demo_dataset = NioVisualLoopDemoDataset(input_json, args.session_dir)
    print('Valid raw loop pairs:', len(demo_dataset), 'All session includes full camera infos:', demo_dataset.all_session_with_cam)
    if len(demo_dataset) != 0 and demo_dataset.all_session_with_cam:
        demo_dataloader = DataLoader(demo_dataset, batch_size=2, shuffle=False, collate_fn=SimpleSingleCollateFnPackMode(), num_workers=16, drop_last=False)
        keyframe_path_list = demo_dataset.keyframe_list
        for iteration, data_dict in tqdm(enumerate(demo_dataloader)):
            out_dict = release_cuda(model(to_cuda(data_dict)))
            loop_descs = out_dict['loop_desc']
            bevs = out_dict['bev']
            for i in range(loop_descs.shape[0]):
                preserved_descs[keyframe_path_list[i + idy]] = loop_descs[i]
                if args.with_yaw:
                    preserved_bevs[keyframe_path_list[i + idy]] = bevs[i]
            idy += loop_descs.shape[0]

    for query_path in tqdm(input_json['cloud_pair_list']):
        idx += 1
        raw_loops = input_json['cloud_pair_list'][query_path]
        query_pcd_loaded = False
        if query_path in preserved_descs.keys():
            query_desc =  preserved_descs[query_path]
            if args.with_yaw:
                query_pose_desc =  preserved_bevs[query_path]
        else:
            continue
        
        query_session_name = query_path.split('/')[0]
        query_pose = np.array(input_json[query_session_name + '_pose_list'][query_path]['T_lidar'])
        query_out_json = {}
        query_out_list = []
        loop_thresh_list = []
        loop_session_list = []
        out_name = query_path.split('/')[-1].replace('bin', 'json')

        base_descs, base_poses, base_pose_descs = [], [], []
        for raw_loop_path in raw_loops:
            if raw_loop_path in preserved_descs.keys():
                base_desc = preserved_descs[raw_loop_path]
                if args.with_yaw:
                    base_pose_desc = preserved_bevs[raw_loop_path]
            else:
                continue
            
            base_session_name = raw_loop_path.split('/')[0]
            base_pose = np.array(input_json[base_session_name + '_pose_list'][raw_loop_path]['T_lidar'])

            desc_dist = np.linalg.norm(query_desc - base_desc)
            # only write predicted loop closures
            if desc_dist > loop_thresh:
                continue
            # compute relative pose of two frames
            if args.with_yaw:
                prior_T = np.matmul(inverse_transform(query_pose), base_pose)
                prior_yaw = rotation_to_euler(get_rotation_translation_from_transform(prior_T)[0])[0]
                pose_input_dict = {'pose_desc_1': torch.from_numpy(query_pose_desc).float(), 'pose_desc_2': torch.from_numpy(base_pose_desc).float()}# load descriptor for pose estimation
                pose_input_dict['prior_yaw'] = prior_yaw
                pose_out_dict = release_cuda(model(to_cuda(pose_input_dict), localizing=True)) # 0.0670 get pose estimation results
                rel_yaw = pose_out_dict['rel_yaw']
                cost = pose_out_dict['min_costs']
                rel_t = pose_out_dict['rel_t'].squeeze()
                if cost > match_thresh:
                    continue

                # registration using gicp
                rel_yaw_in_matrix = euler_to_rotation([rel_yaw, 0, 0])
                rel_yaw_in_transform = np.eye(4)
                rel_yaw_in_transform[:3, :3] = rel_yaw_in_matrix
                rel_yaw_in_transform[2, 3] = rel_t[0]
                rel_yaw_in_transform[1, 3] = rel_t[1]
                if not query_pcd_loaded:
                    if query_path in preserved_lidars.keys():
                        query_pcd = preserved_lidars[query_path]
                    else:
                        query_pcd = get_pointcloud(args.session_dir, query_path, input_json=input_json, o3d_format=True) # 6.1024 load point cloud !!!
                        preserved_lidars[query_path] = query_pcd
                    query, query_tree = small_gicp.preprocess_points(points_numpy=np.array(query_pcd.points), downsampling_resolution=0.1) # 0.22411 preprocess point cloud 
                    query_pcd_loaded = True
                
                if raw_loop_path in preserved_lidars.keys():
                    base_pcd = preserved_lidars[raw_loop_path]
                else:
                    base_pcd = get_pointcloud(args.session_dir, raw_loop_path, input_json=input_json, o3d_format=True)
                    preserved_lidars[raw_loop_path] = base_pcd
                # o3d.io.write_point_cloud("./query.pcd", query_pcd)
                # o3d.io.write_point_cloud("./base.pcd", base_pcd)
                base_pcd.transform(rel_yaw_in_transform)
                # o3d.io.write_point_cloud("./base_trans.pcd", base_pcd)
                base, base_tree = small_gicp.preprocess_points(points_numpy=np.array(base_pcd.points), downsampling_resolution=0.1)
                result_c = small_gicp.align(query, base, query_tree) # 0.6255 estimate fine pose
                error = result_c.error
                if use_dual_reg:
                    base_pcd.transform(result_c.T_target_source)
                    base, base_tree = small_gicp.preprocess_points(points_numpy=np.array(base_pcd.points), downsampling_resolution=0.1)
                    result_f = small_gicp.align(query, base, query_tree)
                    error = result_f.error
                if use_dual_reg:
                    T_query_to_base = np.linalg.inv(result_f.T_target_source) @ np.linalg.inv(result_c.T_target_source) @ np.linalg.inv(rel_yaw_in_transform)
                else:
                    T_query_to_base = np.linalg.inv(result_c.T_target_source) @ np.linalg.inv(rel_yaw_in_transform)
                rel_euler = rotation_to_euler(get_rotation_translation_from_transform(T_query_to_base)[0])
                rel_pitch, rel_roll = rel_euler[1], rel_euler[2]
                # o3d.io.write_point_cloud("./query_trans.pcd", copy.deepcopy(query_pcd).transform(T_query_to_base))
                # print(rel_yaw, prior_yaw, error)
                # if abs(rel_yaw) > 50:
                #     pdb.set_trace()
                if error > reg_thresh or abs(T_query_to_base[2, 3]) > delta_z_thresh or abs(rel_pitch) > p_thresh or abs(rel_roll) > r_thresh:
                    continue

            if  '2' in args.output_style:
                RR, Rt = compute_registration_error(query_pose, base_pose)
                query_out_json[raw_loop_path] = {}
                query_out_json[raw_loop_path]['session_name'] = base_session_name
                query_out_json[raw_loop_path]['keyframe_index'] = int(raw_loop_path.split('/')[-1].split('_')[-1].replace('.bin', ''))
                if args.with_yaw:
                    # RR_est, Rt_est = compute_registration_error(np.linalg.inv(query_pose)@base_pose, T_query_to_base)
                    # query_out_json[raw_loop_path]['estimated_yaw'] = str(rel_yaw)
                    query_out_json[raw_loop_path]['loop_T_cur'] = (T_query_to_base).flatten().tolist()
                    query_out_json[raw_loop_path]['desc_dist'] = float(desc_dist)
                    query_out_json[raw_loop_path]['match_cost'] = float(cost)
                    query_out_json[raw_loop_path]['reg_cost'] = float(error)
                    # for avp mapping
                    query_out_json[raw_loop_path]['fitness_score'] = 0.1
                    query_out_json[raw_loop_path]['overlap_ratio'] = 0.2
                    query_out_json[raw_loop_path]['inner_ratio'] = 0.4
                else:
                    query_out_json[raw_loop_path]['loop_T_cur'] = (np.linalg.inv(query_pose) @ base_pose).flatten().tolist()
                    query_out_json[raw_loop_path]['desc_dist'] = float(desc_dist)
                    query_out_json[raw_loop_path]['Rot_diff'] = float(RR)
                    query_out_json[raw_loop_path]['Tra_diff'] = float(Rt)
                    query_out_json[raw_loop_path]['Z_diff'] = float(abs(query_pose[2, 3] - base_pose[2, 3]))

                if len(query_out_json) > 0:
                    if not os.path.exists(os.path.join(out_path, query_session_name, 'model_infer_result', 'visual_matcher')):
                        make_dir(os.path.join(out_path, query_session_name, 'model_infer_result', 'visual_matcher'))
                    with open(os.path.join(out_path, query_session_name, 'model_infer_result', 'visual_matcher', out_name), 'w') as f:
                        json.dump(query_out_json, f, indent=4)

            if  '1' in args.output_style:
                query_out_list.append(raw_loop_path)
                loop_thresh_list.append(float(desc_dist))
                loop_session_list.append(base_session_name)
            
        if '1' in args.output_style:
            if len(query_out_list) > 0:
                out_json['cloud_pair_list'][query_path] = query_out_list
                loop_thresh_dict[query_path] = loop_thresh_list
                loop_session_dict[query_path] = loop_session_list

    if  '1' in args.output_style:
        # for data not support visual loop closure
        sample_key = True
        if len(out_json['cloud_pair_list']) == 0:
            print('Not support visual loop closure, back to LiDAR-based Loop Closure!')
            sample_key = False
            if os.path.exists(args.input_json.replace('cloud_filter.json', 'cloud_filter_raw.json')):
                raw_input_json_path = args.input_json.replace('cloud_filter.json', 'cloud_filter_raw.json')
                fr =  open(raw_input_json_path)
                raw_input_json = json.load(fr)
                out_json['cloud_pair_list'] = raw_input_json['cloud_pair_list']
            else:
                out_json['cloud_pair_list'] = input_json['cloud_pair_list']
                out_json['cloud_pair_list'] = sample_dict(out_json['cloud_pair_list'], pre_filtered=True)

        # save all  keyframe pose
        for pose_list_key in input_json.keys():
            if pose_list_key != 'cloud_pair_list':
                out_json[pose_list_key] = input_json[pose_list_key]
        if sample_key:
            out_json['cloud_pair_list'] = sample_dict(out_json['cloud_pair_list'], thresh_dict=loop_thresh_dict, session_dict=loop_session_dict)

        if args.debug:
            out_name = './online_infer/cloud_filter_visual_filter_' +  args.input_json.split('_')[-1]
        else:
            out_name = os.path.join(args.result_dir, 'cloud_filter.json')
        with open(out_name, 'w') as f:
            json.dump(out_json, f, indent=4)
        with open(out_name.replace('cloud_filter.json', 'cloud_filter_visual_loop.json'), 'w') as f:
            json.dump(out_json, f, indent=4)

if __name__ == "__main__":

    try:
        main()
    except RuntimeError as e:
        error_message = str(e)
        if "out of memory" in error_message:
            traceback.print_exc()
            logger.error(e)
            sys.exit(100)
        else:
            raise e
