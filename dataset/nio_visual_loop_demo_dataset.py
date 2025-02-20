import os
import re
import cv2
import math
import torch
import random
import numpy as np

import sys

from tqdm import tqdm
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from utils.fisheye import load_fisheye_json, fast_fisheye_distortion
from utils.io_tools import read_json_from_file, read_csv,  read_txt, make_dir, save_dict, load_pkl, read_json_from_file, load_pkl
from utils.geometry import quaternion_to_rotation, rotation_to_quaternion, rotation_to_euler, get_transform_from_rotation_translation, inverse_transform, get_rotation_translation_from_transform, apply_transform

import pdb

def scale_intrinsics(K, sx, sy):
    pose_aug = np.eye(3)
    pose_aug[0, 0] = sx
    pose_aug[1, 1] = sy
    K = pose_aug @ K
    return K

def sample_dict_items(dictionary, num_samples):
    if num_samples == -1:
        return dictionary
    keys = list(dictionary.keys()) 
    sampled_keys = random.sample(keys, min(num_samples, len(keys)))
    sampled_items = {key: dictionary[key] for key in sampled_keys} 
    return sampled_items

class NioVisualLoopDemoDataset(Dataset):
    def __init__(self, input_json, session_dir, img_norm=True):
        self.input_json = input_json
        self.session_dir = session_dir
        self.img_norm = img_norm
        self.json_to_input(input_json)

        if self.img_norm:
            t = [transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            self.transform = transforms.Compose(t)

    def json_to_input(self, input_json):
        keyframe_list = []
        valid_session_list = []
        all_session_list = []
        cloud_pair_list = input_json['cloud_pair_list']
        for query_key_frame in cloud_pair_list.keys():
            query_session_name = query_key_frame.split('/')[0]
            if query_session_name not in all_session_list:
                all_session_list.append(query_session_name)
            if query_key_frame not in keyframe_list:
                if self.check_integrity(self.session_dir, query_key_frame):
                    keyframe_list.append(query_key_frame)
                    if query_session_name not in valid_session_list:
                        valid_session_list.append(query_session_name)
            for base_key_frame in cloud_pair_list[query_key_frame]:
                base_session_name = base_key_frame.split('/')[0]
                if base_session_name not in all_session_list:
                    all_session_list.append(base_session_name)
                if base_key_frame not in keyframe_list:
                    if self.check_integrity(self.session_dir, base_key_frame):
                        keyframe_list.append(base_key_frame)
                        if base_session_name not in valid_session_list:
                            valid_session_list.append(base_session_name)          
        self.keyframe_list = keyframe_list
        if set(valid_session_list) == set(all_session_list):
            self.all_session_with_cam = True
        else:
            self.all_session_with_cam = False

    def check_integrity(self, session_dir, bin_path):
        path_info = bin_path.split('/')
        image_path = os.path.join(session_dir, path_info[0], path_info[1], 'images')
        parameter_path = os.path.join(session_dir, path_info[0], path_info[1], 'parameters', 'camera')
        keyframe_name = path_info[-1].replace('.bin', '')

        fw_img_path = os.path.join(image_path, 'FW', keyframe_name + '.jpg')
        rn_img_path = os.path.join(image_path, 'RN', keyframe_name + '.jpg')
        svc_left_img_path = os.path.join(image_path, 'SVC-Left', keyframe_name + '.jpg')
        svc_right_img_path = os.path.join(image_path, 'SVC-Right', keyframe_name + '.jpg')

        fw_camera_parameter_path = os.path.join(parameter_path, 'front_wide/front_wide.json')
        rn_camera_parameter_path = os.path.join(parameter_path, 'rear_narrow/rear_narrow.json')
        svc_left_camera_parameter_path = os.path.join(parameter_path, 'svc_left/svc_left.json')
        svc_right_camera_parameter_path = os.path.join(parameter_path, 'svc_right/svc_right.json')

        if not os.path.exists(fw_img_path):
            return False
        if not os.path.exists(rn_img_path):
            return False
        if not os.path.exists(svc_left_img_path):
            return False
        if not os.path.exists(svc_right_img_path):
            return False
        if not os.path.exists(fw_camera_parameter_path):
            return False
        if not os.path.exists(rn_camera_parameter_path):
            return False
        if not os.path.exists(svc_left_camera_parameter_path):
            return False
        if not os.path.exists(svc_right_camera_parameter_path):
            return False
        return True

    def load_data(self, session_dir, bin_path, img_height=256, img_width=448):
        # gert necesaary data path from lidar path
        path_info = bin_path.split('/')
        image_path = os.path.join(session_dir, path_info[0], path_info[1], 'images')
        parameter_path = os.path.join(session_dir, path_info[0], path_info[1], 'parameters', 'camera')
        keyframe_name = path_info[-1].replace('.bin', '')

        fw_img_path = os.path.join(image_path, 'FW', keyframe_name + '.jpg')
        rn_img_path = os.path.join(image_path, 'RN', keyframe_name + '.jpg')
        svc_left_img_path = os.path.join(image_path, 'SVC-Left', keyframe_name + '.jpg')
        svc_right_img_path = os.path.join(image_path, 'SVC-Right', keyframe_name + '.jpg')

        fw_camera_parameter_path = os.path.join(parameter_path, 'front_wide/front_wide.json')
        rn_camera_parameter_path = os.path.join(parameter_path, 'rear_narrow/rear_narrow.json')
        svc_left_camera_parameter_path = os.path.join(parameter_path, 'svc_left/svc_left.json')
        svc_right_camera_parameter_path = os.path.join(parameter_path, 'svc_right/svc_right.json')
        
        # load raw images
        fw_img = cv2.imread(fw_img_path)
        rn_img = cv2.imread(rn_img_path)
        fw_img = cv2.resize(fw_img, (img_width, img_height))
        rn_img = cv2.resize(rn_img, (img_width, img_height))

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

        # img normalization
        if not self.img_norm:
            svc_right_img = np.transpose(svc_right_img, [2, 0, 1])
            svc_left_img = np.transpose(svc_left_img, [2, 0, 1])
            fw_img = np.transpose(fw_img, [2, 0, 1])
            rn_img = np.transpose(rn_img, [2, 0, 1])

        # stack camera parameters and imgs
        imgs = np.stack([fw_img, rn_img, svc_left_img, svc_right_img])
        P = np.stack([fw_trans_veh2cam_, rn_trans_veh2cam_, svc_left_trans_veh2cam, svc_right_trans_veh2cam])
        K = np.stack([fw_K, rn_K, svc_left_K, svc_right_K])

        if self.img_norm:
            imgs = [self.transform(e) for e in imgs]

        return {'query_img': imgs,
                'query_K': K,
                'query_P': P,
            } 

    def __len__(self):
        return len(self.keyframe_list)

    def __getitem__(self, idx):
        bin_path = self.keyframe_list[idx]
        data = self.load_data(self.session_dir, bin_path)
        return data
