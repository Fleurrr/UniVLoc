import os
import cv2
import copy
import pickle
import torch
import random
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset
from torchvision import transforms as transforms

import pdb
import sys
sys.path.append("../") 

from prnet.utils.params import TrainingParams
from prnet.datasets.nclt.nclt_raw import NCLTPointCloudLoader, NCLTPointCloudWithImageLoader, NCLTImageLoader
from prnet.datasets.oxford.oxford_raw import OxfordPointCloudLoader, OxfordPointCloudWithImageLoader, OxfordImageLoader
from prnet.utils.data_utils.point_clouds import PointCloudLoader, PointCloudWithImageLoader
from prnet.utils.data_utils.poses import m2ypr
from prnet.utils.data_utils.poses import apply_transform
from prnet.datasets.panorama import generate_sph_image
from prnet.datasets.augmentation import TrainTransform, TrainSetTransform, TrainRGBTransform, ValRGBTransform

from utils.io_tools import read_json_from_file, read_csv,  read_txt, make_dir, save_dict, load_pkl, read_json_from_file
from utils.geometry import quaternion_to_rotation, rotation_to_quaternion, rotation_to_euler, get_transform_from_rotation_translation, inverse_transform, get_rotation_translation_from_transform, apply_transform

def scale_intrinsics(K, sx, sy):
    pose_aug = np.eye(3)
    pose_aug[0, 0] = sx
    pose_aug[1, 1] = sy
    K = pose_aug @ K
    return K

class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, pose: np, positives_poses: Dict[int, np.ndarray] = None, filepaths: list = None):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # pose: pose as 4x4 matrix
        # positives_poses: relative poses of positive examples refined using ICP
        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.pose = pose
        self.positives_poses = positives_poses
        self.filepaths = filepaths

class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array, pose: np.array = None, filepaths: list = None):
        # position: x, y position in meters
        # pose: 6 DoF pose (as 4x4 pose matrix)
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position
        self.pose = pose
        self.filepaths = filepaths

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position, self.pose, self.filepaths

class  OxfordVisualLoopDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_type: str, query_filename: str, transform=None, set_transform=TrainTransform(3), image_transform=TrainRGBTransform(2), params=None, neg_mine=True):
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        print("dataset type ", dataset_type)
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.set_transform = set_transform
        self.image_transform = image_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        print('{} queries in the dataset'.format(len(self)))

        self.unused_infos = copy.deepcopy(self.queries)

        self.params = params
        if self.dataset_type == 'oxford':
            self.extrinsics_dir = os.path.join('/home/g.huang/visual_loop_closure/dataset/Oxford', 'extrinsics')
        else:
            self.extrinsics_dir = None

        # pc_loader must be set in the inheriting class
        self.im_loader = get_image_loader(self.dataset_type)
        self.width = 448
        self.height = 256

        # camera parameter path
        image_meta_path = '/map-hl/gary.huang/public_dataset/Oxford/dataset/image_meta.pkl'
        self.image_meta = load_pkl(image_meta_path)
        self.K = np.array(self.image_meta['K'])
        self.K = scale_intrinsics(self.K, self.width / 640, self.height/320)
        self.P = self.image_meta['T']
        if self.transform is not None:
            self.P = np.array([self.extrinsics_transform(e) for e in self.P])
        else:
            self.P = np.array(self.P)

        self.neg_mine = neg_mine
        self.neg_list = []

    def __len__(self):
        return len(self.queries)

    def update_neg_list(self, hard_idxs):
        for i in range(hard_idxs.shape[0]):
            hard_info = str(int(hard_idxs[i, 0])) + '_' + str(int(hard_idxs[i, 1])) + '_' + str(int(hard_idxs[i, 2]))
            if hard_info not in self.neg_list:
                self.neg_list.append(hard_info)

    def extrinsics_transform(self, pose_matrix):
        R_c = pose_matrix[:3, :3]
        t_c = pose_matrix[:3, 3]

        R_c_new = R_c @ self.transform
        t_c_new = t_c

        new_pose_matrix = np.eye(4)
        new_pose_matrix[:3, :3] = R_c_new
        new_pose_matrix[:3, 3] = t_c_new.ravel()
        return new_pose_matrix

    def pose_transform(self, pose_matrix):
        R = np.eye(4)
        R[:3, :3] = self.transform
        new_pose_matrix = R @ pose_matrix @ R.T
        return new_pose_matrix

    def get_data(self, ndx):
        # Load point cloud and apply transform
        if self.params.model_params.use_panorama:
            sph = True
        else:
            sph = False

        if self.dataset_type == 'oxford':
            query_imgs = self.im_loader(self.queries[ndx].filepaths, sph, self.extrinsics_dir)
        else:
            file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
            query_imgs = self.im_loader(file_pathname, sph)
        
        query_imgs = [cv2.resize(e, [self.width, self.height]) for e in query_imgs]

        if self.image_transform is not None:
            query_imgs = [self.image_transform(e) for e in query_imgs]
        else:
            t = [transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            transform = transforms.Compose(t)
            query_imgs = [transform(e) for e in query_imgs]

        query_imgs = np.stack(query_imgs)

        if self.transform is not None:
            return query_imgs, ndx, self.pose_transform(self.queries[ndx].pose)  #(N, 3), (S, 3, H, W), index,
        else:
            return query_imgs, ndx, self.queries[ndx].pose  #(N, 3), (S, 3, H, W), index,

    def __getitem__(self, ndx):
        rdn_pos_idx, rdn_neg_idx = None, None
        use_neg_mine = False
        if self.neg_mine and len(self.neg_list) > 8 and random.random() > 0.4:
            hard_qry_idx = random.randint(0, len(self.neg_list) - 1)
            hard_info = self.neg_list[hard_qry_idx].split('_')
            _ = self.neg_list.pop(hard_qry_idx)
            ndx, rdn_pos_idx, rdn_neg_idx = int(hard_info[0]), int(hard_info[1]), int(hard_info[2])
            use_neg_mine = True

        # get data
        query_imgs, query_index, query_T = self.get_data(ndx)
        positive_indexes = self.get_positives(query_index)
        if rdn_pos_idx is None:
            rdn_pos_idx = random.randint(0, positive_indexes.shape[0] - 1)
        positive_imgs, positive_index, positive_T = self.get_data(positive_indexes[rdn_pos_idx])
        non_negative_indexes = self.get_non_negatives(query_index)
        negative_indexes = np.setdiff1d(np.arange(len(self.queries)), non_negative_indexes)
        if rdn_neg_idx is None:
            rdn_neg_idx = random.randint(0, negative_indexes.shape[0] - 1)
        negative_imgs, negative_index, negative_T = self.get_data(negative_indexes[rdn_neg_idx])

        # update unused ndx
        # if not use_neg_mine:
        #     self.update_unused_info(ndx, rdn_pos_idx, rdn_neg_idx)
        
        # get relative poses
        rel_T = np.matmul(inverse_transform(query_T), positive_T)
        rel_R, rel_t = get_rotation_translation_from_transform(rel_T)
        rel_q = rotation_to_quaternion(rel_R)
        rel_euler = rotation_to_euler(rel_R)

        # save random sample indexes:
        rdn_idx = np.ones(3)
        rdn_idx[0] = ndx
        rdn_idx[1] = rdn_pos_idx
        rdn_idx[2] = rdn_neg_idx

        data_dict =  {
                       'query_imgs': query_imgs,
                       'positive_imgs':  positive_imgs,
                       'negative_imgs': negative_imgs[np.newaxis, ...], #(N, S, C, H, W)
                       'query_K': self.K,
                       'positive_K': self.K,
                       'negative_K': self.K[np.newaxis, ...],
                       'query_P': self.P,
                       'positive_P': self.P,
                       'negative_P': self.P[np.newaxis, ...],
                       'relative_rotation': rel_R,
                       'relative_translation': rel_t,
                       'relative_quaternion': rel_q,
                       'relative_euler': rel_euler,
                       'rdn_idx': rdn_idx,
        }
        return data_dict

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

    def update_unused_info(self, ndx, rdn_pos_idx, rdn_neg_idx):
        self.unused_infos[ndx].positives = np.array(list(self.unused_infos[ndx].positives).pop(rdn_pos_idx))
        self.unused_infos[ndx].non_negatives = np.array(list(self.unused_infos[ndx].non_negatives).append(rdn_neg_idx))
        if self.unused_infos[ndx].positives.shape[0] == 0:
             self.unused_infos[ndx].positives = copy.deepcopy(self.queires[ndx].positives)
        if self.unused_infos[ndx].non_negatives.shape[0] >= (len(self.queires) - 1):
            self.unused_infos[ndx].non_negatives = copy.deepcopy(self.queires[ndx].non_negatives)

class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set
        self.transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def pose_transform(self, pose_matrix):
        R = np.eye(4)
        R[:3, :3] = self.transform
        new_pose_matrix = R @ pose_matrix @ R.T
        return new_pose_matrix

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            if len(e) == 5:
                self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], e[4]))
            else:
                self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], None))

        self.map_set = []
        for e in map_l:
            if len(e) == 5:
                self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], e[4]))
            else:
                self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], None))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            if self.transform is not None:
                positions[ndx] = self.pose_transform(pos.pose)[:2, 3]
            else:
                positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            if self.transform is not None:
                positions[ndx] = self.pose_transform(pos.pose)[:2, 3]
            else:
                positions[ndx] = pos.position
        return positions

    def get_map_poses(self):
        # Get map positions as (N, 2) array
        poses = np.zeros((len(self.map_set), 4, 4), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            if self.transform is not None:
                poses[ndx] = self.pose_transform(pos.pose)
            else:
                poses[ndx] = pos.pose
        return poses

    def get_query_poses(self):
        # Get query positions as (N, 2) array
        poses = np.zeros((len(self.query_set), 4, 4), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            if self.transform is not None:
                poses[ndx] = self.pose_transform(pos.pose)
            else:
                poses[ndx] = pos.pose
        return poses

def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    if dataset_type == 'nclt':
        return NCLTPointCloudLoader()
    elif dataset_type == 'oxford':
        return OxfordPointCloudLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")

def get_pointcloud_with_image_loader(dataset_type) -> PointCloudWithImageLoader:
    if dataset_type == 'nclt':
        return NCLTPointCloudWithImageLoader()
    elif dataset_type == 'oxford':
        return OxfordPointCloudWithImageLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")

def get_image_loader(dataset_type) -> PointCloudWithImageLoader:
    if dataset_type == 'nclt':
        return NCLTImageLoader()
    elif dataset_type == 'oxford':
        return OxfordImageLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")

if __name__ == "__main__":
    params = TrainingParams('../prnet/config/deformable/config_deformable.txt', '../prnet/config/deformable/deformable.txt')
    params.print()
    dataset = OxfordVisualLoopDataset(dataset_path='/map-hl/gary.huang/public_dataset/Oxford/dataset/',  \
                                                            dataset_type='oxford', query_filename='train_2019-01-11-13-24-51-radar-oxford-10k_2019-01-15-13-06-37-radar-oxford-10k_2.0_3.0.pickle', params=params)
    data = dataset[1]
    pdb.set_trace()
