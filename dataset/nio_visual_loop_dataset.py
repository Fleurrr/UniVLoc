import os
import re
import cv2
import math
import torch
import random
import argparse
import numpy as np

import sys
sys.path.append("../") 

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

class NioVisualLoopDataset(Dataset):
    def __init__(self, dataset_path, with_amb=False,
                              img_width=427, img_height=240, sample_query=5,  img_norm=True, 
                              keyframe_thresh=5, positive_thresh=2., negative_min_thresh=5., negative_max_thresh=50., floor_height=3, 
                              dataset_length=-1, neg_num=2, total_sample_limit=700,
                              data_info_path='./data/data_info.txt', loop_closure_path='./loop_closure.pkl', mode='train', demo=False, neg_mine=True):
        self.dataset_path = dataset_path
        self.mode = mode
        self.demo = demo
        self.keyframe_thresh = keyframe_thresh
        self.positive_thresh = positive_thresh
        self.negative_min_thresh = negative_min_thresh
        self.negative_max_thresh = negative_max_thresh
        self.floor_height = floor_height

        self.img_height = img_height
        self.img_width = img_width
        self.img_norm = img_norm
        if self.img_norm:
            t = [transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            self.transform = transforms.Compose(t)

        self.dataset_length = dataset_length
        self.neg_num = neg_num
        self.sample_limit = total_sample_limit
        self.sample_query = sample_query
        self.with_amb = with_amb

        self.neg_mine = neg_mine
        self.neg_list = []

        self.info = {}
        if data_info_path is not None:
            with open(data_info_path, 'r') as file:
                for line in file:
                    line_info = line.strip().split(' ')
                    self.info[line_info[0]] = [int(line_info[1]), int(line_info[2])]

        self.loop_closure_path = loop_closure_path
        if os.path.exists(self.loop_closure_path):
            self.loop_closures =  load_pkl(self.loop_closure_path)
        else:
            self.loop_closures = self.prepare_loop_closures()
            pickle_name = self.loop_closure_path.split('/')[-1].replace('.pkl', '')
            save_dict('./', self.loop_closures, pickle_name, add=True)

    def prepare_loop_closures(self):
        loop_closures = {}
        for place_name in tqdm(os.listdir(self.dataset_path)):
            if len(self.info) != 0:
                if place_name in self.info:
                    # if self.info[place_name][1] > 0: # outdoor and session number > 1
                    #     continue
                    # if self.info[place_name][0] < 1 or self.info[place_name][1] == 0: # indoor and session number > 1
                    
                    if self.info[place_name][1] > 0: # indoor, tighten thresh
                        self.negative_min_thresh = 3.

            if os.path.exists(os.path.join(self.dataset_path, place_name, 'others')):  
                sessions_dir = os.path.join(self.dataset_path, place_name, 'others/sessions')
            elif os.path.exists(os.path.join(self.dataset_path, place_name, 'sessions')):
                sessions_dir = os.path.join(self.dataset_path, place_name, 'sessions')
            elif os.path.exists(os.path.join(self.dataset_path, place_name, 'sessions_data')):
                sessions_dir = os.path.join(self.dataset_path, place_name, 'sessions_data')
            # need to merge all session information
            FW_info, RN_info, SVC_Left_info, SVC_Right_info = {}, {}, {}, {}
            start = 0
            
            for session_name in os.listdir(sessions_dir):
                if session_name.endswith('.pkl'):
                    continue

                FW_info_path = os.path.join(sessions_dir, session_name + '_FW_info.pkl')
                if not os.path.exists(FW_info_path):
                    break
                RN_info_path = os.path.join(sessions_dir, session_name + '_RN_info.pkl')
                if not os.path.exists(RN_info_path):
                    break
                SVC_Left_info_path = os.path.join(sessions_dir, session_name + '_SVC_Left_info.pkl')
                if not os.path.exists(SVC_Left_info_path):
                    break
                SVC_Right_info_path = os.path.join(sessions_dir, session_name + '_SVC_Right_info.pkl')
                if not os.path.exists(SVC_Right_info_path):
                    break
                # session_path = os.path.join(sessions_dir, session_name)
                
                FW_info_single_session = load_pkl(FW_info_path)
                RN_info_single_session = load_pkl(RN_info_path)
                SVC_Left_info_single_session = load_pkl(SVC_Left_info_path)
                SVC_Right_info_single_session = load_pkl(SVC_Right_info_path)
                
                # merge into one dict
                for i, key in enumerate(FW_info_single_session):
                    FW_info[key + start] = FW_info_single_session[key]
                fw_key_max = key
                for i, key in enumerate(RN_info_single_session):
                    RN_info[key + start] = RN_info_single_session[key]
                rn_key_max = key
                for i, key in enumerate(SVC_Left_info_single_session):
                    SVC_Left_info[key + start] = SVC_Left_info_single_session[key]
                svc_left_key_max = key
                for i, key in enumerate(SVC_Right_info_single_session):
                    SVC_Right_info[key + start] = SVC_Right_info_single_session[key]
                svc_right_key_max = key
                start = max(fw_key_max, rn_key_max, svc_left_key_max, svc_right_key_max)
                
            # serve FW as the main camera
            for i, q_key in enumerate(FW_info):
                if i % self.sample_query != 0:
                    continue
                query_cord = FW_info[q_key][-1][:3]
                positive_id, negative_id = {}, {}
                if self.with_amb:
                    ambiguous_id = {}
                if q_key not in RN_info.keys() or q_key not in FW_info.keys() or q_key not in SVC_Left_info.keys() or q_key not in SVC_Right_info.keys():
                    continue
                
                for j, d_key in enumerate(FW_info):
                    database_cord = FW_info[d_key][-1][:3]
                    query_cord = np.array(query_cord)
                    database_cord = np.array(database_cord)
                    dist = np.linalg.norm(query_cord  - database_cord)
                    
                    detailed_info_dict = {}
                    if abs(q_key - d_key) <= self.keyframe_thresh:
                        continue
                    elif dist < self.positive_thresh and abs(query_cord[2] - database_cord[2]) < self.floor_height:
                            if d_key not in RN_info.keys() or d_key not in FW_info.keys() or d_key not in SVC_Left_info.keys() or d_key not in SVC_Right_info.keys():
                                continue
                            # processed data info, format: {keyframe id: ts, name, path, [pose]}
                            detailed_info_dict['pose'] = FW_info[d_key][-1]
                            detailed_info_dict['fw'] = os.path.join(FW_info[d_key][2], FW_info[d_key][1])
                            detailed_info_dict['rn'] = os.path.join(RN_info[d_key][2], RN_info[d_key][1])
                            detailed_info_dict['svc_left'] = os.path.join(SVC_Left_info[d_key][2], SVC_Left_info[d_key][1])
                            detailed_info_dict['svc_right'] = os.path.join(SVC_Right_info[d_key][2], SVC_Right_info[d_key][1])
                             
                            positive_id[d_key] = detailed_info_dict
                    elif  (dist > self.negative_min_thresh and dist < self.negative_max_thresh) or (abs(query_cord[2] - database_cord[2]) > self.floor_height and dist < self.negative_max_thresh):
                            if d_key not in RN_info.keys() or d_key not in FW_info.keys() or d_key not in SVC_Left_info.keys() or d_key not in SVC_Right_info.keys():
                                continue
                            detailed_info_dict['pose'] = FW_info[d_key][-1]
                            detailed_info_dict['fw'] = os.path.join(FW_info[d_key][2], FW_info[d_key][1])
                            detailed_info_dict['rn'] = os.path.join(RN_info[d_key][2], RN_info[d_key][1])
                            detailed_info_dict['svc_left'] = os.path.join(SVC_Left_info[d_key][2], SVC_Left_info[d_key][1])
                            detailed_info_dict['svc_right'] = os.path.join(SVC_Right_info[d_key][2], SVC_Right_info[d_key][1])
                            
                            negative_id[d_key] = detailed_info_dict

                    elif  self.with_amb and (dist > self.positive_thresh and dist < self.negative_min_thresh) and (abs(query_cord[2] - database_cord[2]) < self.floor_height):
                            if d_key not in RN_info.keys() or d_key not in FW_info.keys() or d_key not in SVC_Left_info.keys() or d_key not in SVC_Right_info.keys():
                                continue
                            detailed_info_dict['pose'] = FW_info[d_key][-1]
                            detailed_info_dict['fw'] = os.path.join(FW_info[d_key][2], FW_info[d_key][1])
                            detailed_info_dict['rn'] = os.path.join(RN_info[d_key][2], RN_info[d_key][1])
                            detailed_info_dict['svc_left'] = os.path.join(SVC_Left_info[d_key][2], SVC_Left_info[d_key][1])
                            detailed_info_dict['svc_right'] = os.path.join(SVC_Right_info[d_key][2], SVC_Right_info[d_key][1])
                            
                            ambiguous_id[d_key] = detailed_info_dict                            

                if len(positive_id) > 0:
                    loop_info = {}
                    positive_id = sample_dict_items(positive_id, self.sample_limit)
                    negative_id = sample_dict_items(negative_id, self.sample_limit)
                    loop_info['positive'] = positive_id
                    loop_info['negative'] = negative_id
                    if self.with_amb:
                        ambiguous_id = sample_dict_items(ambiguous_id, self.sample_limit)
                        loop_info['ambiguous'] = ambiguous_id
                    
                    detailed_info_dict = {}
                    detailed_info_dict['pose'] = FW_info[q_key][-1]
                    detailed_info_dict['fw'] = os.path.join(FW_info[q_key][2], FW_info[q_key][1])
                    detailed_info_dict['rn'] = os.path.join(RN_info[q_key][2], RN_info[q_key][1])
                    detailed_info_dict['svc_left'] = os.path.join(SVC_Left_info[q_key][2], SVC_Left_info[q_key][1])
                    detailed_info_dict['svc_right'] = os.path.join(SVC_Right_info[q_key][2], SVC_Right_info[q_key][1])                   
                    loop_info['query'] = detailed_info_dict
                    if self.mode == 'train':
                        loop_closures[place_name + '_' + str(q_key)] = loop_info
                    elif self.mode == 'eval':
                        if not place_name in loop_closures.keys():
                            loop_closures[place_name] = {}
                        loop_closures[place_name][str(q_key)] = loop_info
            
            # if self.mode == 'train':
            #     save_dict('./', loop_closures, 'loop_closure', add=True)
            # elif self.mode == 'eval':
            #    save_dict('./', loop_closures, 'loop_closure_eval', add=True)
        return loop_closures
        
    def load_image_from_data(self, query_data, with_param=True):
        for cam_name in ['fw', 'rn', 'svc_left', 'svc_right']:
            if self.demo:
                query_data[cam_name] = query_data[cam_name].replace('./data/training_1230/', './data/evaling/')
                query_data[cam_name] = query_data[cam_name].replace('./data/training/', './data/evaling/')
            else:
                query_data[cam_name] = query_data[cam_name].replace('./data/training_1230/', './data/training/') 
                query_data[cam_name] = query_data[cam_name].replace('./data/training_0503/', './data/training/') 
                query_data[cam_name] = query_data[cam_name].replace('./data/training_0430/', './data/training/') 
            if self.mode == 'eval':
                query_data[cam_name] = query_data[cam_name].replace('./data/training_1230/', './data/evaling/')
                query_data[cam_name] = query_data[cam_name].replace('./data/training/', './data/evaling/')

        fw_img = cv2.imread(query_data['fw'] + '.jpg')
        rn_img = cv2.imread(query_data['rn'] + '.jpg')

        fw_img = cv2.resize(fw_img, (self.img_width, self.img_height))
        rn_img = cv2.resize(rn_img, (self.img_width, self.img_height))

        if with_param:
            s_index = query_data['fw'].find('image')
            fw_camera_parameter_path = os.path.join(query_data['fw'][:s_index], 'parameters/camera/front_wide/front_wide.json')
            fw_cam_param = read_json_from_file(fw_camera_parameter_path)
            fw_K, fw_Rotation, fw_translation =fw_cam_param['intrinsic_param']['camera_matrix'], fw_cam_param['extrinsic_param']['rotation'], fw_cam_param['extrinsic_param']['translation']
            fw_origin_h, fw_origin_w = fw_cam_param['intrinsic_param']['camera_height'], fw_cam_param['intrinsic_param']['camera_width']

            fw_Rotation = Rotation.from_rotvec(fw_Rotation)
            fw_trans_cam2veh_ = np.eye(4)
            fw_trans_cam2veh_[:3, :3] = fw_Rotation.as_matrix()
            fw_trans_cam2veh_[:3, 3] = fw_translation
            fw_trans_veh2cam_ = np.linalg.inv(fw_trans_cam2veh_)
            # rescale intrinsics
            fw_K = scale_intrinsics(fw_K, self.img_width/fw_origin_w, self.img_height/fw_origin_h)

            s_index = query_data['rn'].find('image')
            rn_camera_parameter_path = os.path.join(query_data['rn'][:s_index], 'parameters/camera/rear_narrow/rear_narrow.json')
            rn_cam_param = read_json_from_file(rn_camera_parameter_path)
            rn_K, rn_Rotation, rn_translation =rn_cam_param['intrinsic_param']['camera_matrix'], rn_cam_param['extrinsic_param']['rotation'], rn_cam_param['extrinsic_param']['translation']
            rn_origin_h, rn_origin_w = fw_cam_param['intrinsic_param']['camera_height'], fw_cam_param['intrinsic_param']['camera_width']

            rn_Rotation = Rotation.from_rotvec(rn_Rotation)
            rn_trans_cam2veh_ = np.eye(4)
            rn_trans_cam2veh_[:3, :3] = rn_Rotation.as_matrix()
            rn_trans_cam2veh_[:3, 3] = rn_translation
            rn_trans_veh2cam_ = np.linalg.inv(rn_trans_cam2veh_)
            # rescale intrinsics
            rn_K = scale_intrinsics(rn_K, self.img_width/rn_origin_w, self.img_height/rn_origin_h)

        svc_left_img = cv2.imread(query_data['svc_left'] + '.jpg')
        s_index = query_data['svc_left'].find('image')
        svc_left_camera_parameter_path = os.path.join(query_data['svc_left'][:s_index], 'parameters/camera/svc_left/svc_left.json')
        svc_left_cam_param = load_fisheye_json(svc_left_camera_parameter_path, 'SVC-Left')
        svc_left_img, svc_left_K = fast_fisheye_distortion(svc_left_img, svc_left_cam_param, self.img_width, self.img_height)
        
        svc_right_img = cv2.imread(query_data['svc_right'] + '.jpg')
        s_index = query_data['svc_right'].find('image')
        svc_right_camera_parameter_path = os.path.join(query_data['svc_right'][:s_index], 'parameters/camera/svc_right/svc_right.json')
        svc_right_cam_param = load_fisheye_json(svc_right_camera_parameter_path, 'SVC-Right')
        svc_right_img, svc_right_K = fast_fisheye_distortion(svc_right_img, svc_right_cam_param, self.img_width, self.img_height)
        if with_param:
            svc_left_trans_veh2cam =  svc_left_cam_param['trans_veh2cam_']
            svc_right_trans_veh2cam = svc_right_cam_param['trans_veh2cam_']
        
        if not self.img_norm:
            svc_right_img = np.transpose(svc_right_img, [2, 0, 1])
            svc_left_img = np.transpose(svc_left_img, [2, 0, 1])
            fw_img = np.transpose(fw_img, [2, 0, 1])
            rn_img = np.transpose(rn_img, [2, 0, 1])

        if with_param:
            return [fw_img, rn_img, svc_left_img, svc_right_img],  \
                          [fw_trans_veh2cam_, rn_trans_veh2cam_, svc_left_trans_veh2cam, svc_right_trans_veh2cam ], \
                          [fw_K, rn_K, svc_left_K, svc_right_K]
        else:
            return [fw_img, rn_img, svc_left_img, svc_right_img]
    
    def merge_data_in_list(self, data):
        return np.stack(data)

    def update_neg_list(self, hard_idxs):
        for i in range(hard_idxs.shape[0]):
            hard_info = str(int(hard_idxs[i, 0])) + '_' + str(int(hard_idxs[i, 1])) + '_' + str(int(hard_idxs[i, 2]))
            if hard_info not in self.neg_list:
                self.neg_list.append(hard_info)

    def load_meta_data_training(self, meta_data, qry_idx=0, pos_idx=None, neg_idx=None):
        # load query data
        query_data = meta_data['query']
        query_imgs, query_P, query_K= self.load_image_from_data(query_data, True)
        query_pose = np.array(query_data['pose'])
        query_t, query_R = query_pose[:3], quaternion_to_rotation(query_pose[3:])
        query_T = get_transform_from_rotation_translation(query_R, query_t)
        
        # load positive data
        if self.neg_mine and pos_idx is not None:
            all_positive_data = meta_data['positive']
            positive_data = all_positive_data[list(all_positive_data.keys())[pos_idx]]
            rdn_pos_idx = pos_idx
        else:
            all_positive_data = meta_data['positive']
            rdn_pos_idx = random.randint(0, len(all_positive_data.keys()) - 1)
            positive_data = all_positive_data[list(all_positive_data.keys())[rdn_pos_idx]]
        positive_imgs, positive_P, positive_K = self.load_image_from_data(positive_data, True)
        positive_pose = np.array(positive_data['pose'])
        positive_t, positive_R = positive_pose[:3], quaternion_to_rotation(positive_pose[3:])
        positive_T = get_transform_from_rotation_translation(positive_R, positive_t)
        
        # load negative data
        all_negative_data = meta_data['negative']
        negative_imgs_list, negative_P_list, negative_K_list = [], [], []
        for nn in range(self.neg_num):
            if self.neg_mine and neg_idx is not None and nn == 0:
                negative_data = all_negative_data[list(all_negative_data.keys())[neg_idx]]
                rdn_neg_idx = neg_idx
            else:
                rdn_neg_idx = random.randint(0, len(all_negative_data.keys()) - 1)
                negative_data = all_negative_data[list(all_negative_data.keys())[rdn_neg_idx]]
            negative_imgs, negative_P, negative_K = self.load_image_from_data(negative_data, True)
            negative_pose = np.array(negative_data['pose'])
            negative_t, negative_R = negative_pose[:3], quaternion_to_rotation(negative_pose[3:])
            negative_T = get_transform_from_rotation_translation(negative_R, negative_t)
            if self.img_norm:
                negative_imgs = [self.transform(e) for e in negative_imgs]
            negative_imgs_list.append(self.merge_data_in_list(negative_imgs))
            negative_P_list.append(self.merge_data_in_list(negative_P))
            negative_K_list.append(self.merge_data_in_list(negative_K))

        # load ambiguous data
        if self.with_amb:
            all_ambiguous_data = meta_data['ambiguous']
            ambiguous_imgs_list, ambiguous_P_list, ambiguous_K_list = [], [], []
            rdn_amb_idx = random.randint(0, len(all_ambiguous_data.keys()) - 1)
            ambiguous_data = all_ambiguous_data[list(all_ambiguous_data.keys())[rdn_amb_idx]]
            ambiguous_imgs, ambiguous_P, ambiguous_K = self.load_image_from_data(ambiguous_data, True)
            ambiguous_pose = np.array(ambiguous_data['pose'])
            ambiguous_t, ambiguous_R = ambiguous_pose[:3], quaternion_to_rotation(ambiguous_pose[3:])
            ambiguous_T = get_transform_from_rotation_translation(ambiguous_R, ambiguous_t)
        
        # calculate ground-truth relative poses between query and positive data
        rel_T = np.matmul(inverse_transform(query_T), positive_T)
        rel_R, rel_t = get_rotation_translation_from_transform(rel_T)
        rel_q = rotation_to_quaternion(rel_R)
        rel_euler = rotation_to_euler(rel_R)

        # normalize
        if self.img_norm:
            query_imgs = [self.transform(e) for e in query_imgs]
            positive_imgs = [self.transform(e) for e in positive_imgs]
        
        # save random sample indexes:
        rdn_idx = np.ones(3)
        rdn_idx[0] = qry_idx
        rdn_idx[1] = rdn_pos_idx
        rdn_idx[2] = rdn_neg_idx

        # merge list img to a single img shaped as (4, h, w, 3)
        data_dict =  {'query_imgs': self.merge_data_in_list(query_imgs),
                       'positive_imgs':  self.merge_data_in_list(positive_imgs),
                       'negative_imgs': self.merge_data_in_list(negative_imgs_list), #(N, S, C, H, W)
                       'query_K': self.merge_data_in_list(query_K),
                       'positive_K': self.merge_data_in_list(positive_K),
                       'negative_K': self.merge_data_in_list(negative_K_list),
                       'query_P': self.merge_data_in_list(query_P),
                       'positive_P': self.merge_data_in_list(positive_P),
                       'negative_P': self.merge_data_in_list(negative_P_list),
                       'relative_rotation': rel_R,
                       'relative_translation': rel_t,
                       'relative_quaternion': rel_q,
                       'relative_euler': rel_euler,
                       'rdn_idx': rdn_idx,
        }
        if self.with_amb:
            data_dict['ambiguous_K'] = self.merge_data_in_list(ambiguous_K)
            data_dict['ambiguous_P'] = self.merge_data_in_list(ambiguous_P)
            data_dict['ambiguous_imgs'] = self.merge_data_in_list(ambiguous_imgs)
        return data_dict

    def load_meta_data_evaling(self, meta_data, to_tensor=True, query_only=True):
        # load query data
        if query_only:
            query_data = meta_data['query']
        else:
            query_data = meta_data
        query_imgs, query_P, query_K= self.load_image_from_data(query_data, True)
        query_pose = np.array(query_data['pose'])
        query_t, query_R = query_pose[:3], quaternion_to_rotation(query_pose[3:])
        query_T = get_transform_from_rotation_translation(query_R, query_t)
        if self.img_norm:
            query_imgs = [self.transform(e) for e in query_imgs]
        
        # merge list img to a single img shaped as (4, h, w, 3)
        if to_tensor:
            return {'query_imgs': torch.from_numpy(self.merge_data_in_list(query_imgs)).unsqueeze(0).float(),
                        'query_K': torch.from_numpy(self.merge_data_in_list(query_K)).unsqueeze(0).float(),
                        'query_P': torch.from_numpy(self.merge_data_in_list(query_P)).unsqueeze(0).float(),
                        'query_T': torch.from_numpy(query_T).unsqueeze(0).float(),
            }
        else:
             return {'query_imgs': self.merge_data_in_list(query_imgs),
                        'query_K': self.merge_data_in_list(query_K),
                        'query_P': self.merge_data_in_list(query_P),
                        'query_T': query_T,
            } 

    def __len__(self):
        keys = self.loop_closures.keys()
        origin_length = len(keys)
        if self.dataset_length == -1 or self.mode == 'eval' or origin_length == self.dataset_length:
            return origin_length
        else:
            return self.dataset_length
    
    def get_training_items(self, idx):
        keys = list(self.loop_closures.keys())
        origin_length = len(keys)
        if self.dataset_length == -1 or origin_length == self.dataset_length:
            key = keys[idx]
        elif origin_length > self.dataset_length:
            div = math.ceil(origin_length / self.dataset_length)
            shift = random.randint(0, div - 1)
            idy = int(div * idx + shift)
            if idy > origin_length - 1:
                idy = origin_length - 1
            key = keys[idy]
            idx = idy
        elif origin_length < self.dataset_length:
            idy = idx % origin_length
            key = keys[idy]
            idx = idy

        if self.neg_mine and len(self.neg_list) > 8 and random.random() > 0.4:
            hard_qry_idx = random.randint(0, len(self.neg_list) - 1)
            hard_info = self.neg_list[hard_qry_idx].split('_')
            _ = self.neg_list.pop(hard_qry_idx)
            idx, pos_idx, neg_idx = int(hard_info[0]), int(hard_info[1]), int(hard_info[2])
            key = keys[idx]
            meta_data = self.loop_closures[key]
            return self.load_meta_data_training(meta_data, idx, pos_idx, neg_idx)
        else:
            meta_data = self.loop_closures[key]
            return self.load_meta_data_training(meta_data, idx)
    
    def get_evaling_items(self, idx):
        keys = list(self.loop_closures.keys())
        key = keys[idx]
        meta_datas = self.loop_closures[key]
        return meta_datas

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.get_training_items(idx)
        elif self.mode == 'eval':
            return self.get_evaling_items(idx)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/map-hl/gary.huang/visual_loop_closure_dataset/")
    parser.add_argument("--data_info_path", type=str, default=None)
    args = parser.parse_args()
   # _ = NioVisualLoopDataset(dataset_path='/map-hl/gary.huang/visual_loop_closure_dataset/training',  positive_thresh=2, data_info_path='../data/data_info.txt', loop_closure_path='../dataset/loop_closure_o2_i2_n5_U_l700_test.pkl', dataset_length=-1, mode='train')
    _ = NioVisualLoopDataset(dataset_path=os.path.join(args.dataset_path, 'finetune'),  positive_thresh=2, negative_min_thresh=3, data_info_path=args.data_info_path, loop_closure_path='../dataset/loop_closure_o2_i2_n3_U_l700_finetune.pkl', dataset_length=-1, mode='train')
    _ = NioVisualLoopDataset(dataset_path=os.path.join(args.dataset_path, 'evaling'),  positive_thresh=2, negative_min_thresh=3, data_info_path=args.data_info_path, loop_closure_path='../dataset/loop_closure_finetune_eval.pkl', dataset_length=-1, mode='eval')

    