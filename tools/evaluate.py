# Zhejiang University

import argparse
import numpy as np
import tqdm
import os
import cv2
import random
from typing import List
import open3d as o3d
from time import time
import torch
import torch.nn as nn
from torchvision import transforms as transforms
from prnet.utils.data_utils.poses import m2ypr, relative_pose
from prnet.utils.data_utils.point_clouds import icp, make_open3d_feature, make_open3d_point_cloud
from prnet.utils.data_utils.poses import apply_transform, m2ypr
from prnet.utils.params import TrainingParams, ModelParams
from prnet.datasets.dataset_utils import preprocess_pointcloud
from prnet.datasets.panorama import generate_sph_image
from prnet.utils.loss_utils import *
from prnet.datasets.range_image import range_projection

import pdb
from utils.torch_utils import to_cuda, all_reduce_tensors, release_cuda, initialize
from utils.io_tools import read_json_from_file, read_csv,  read_txt, make_dir, save_dict, load_pkl, read_json_from_file
from utils.geometry import quaternion_to_rotation, rotation_to_quaternion, rotation_to_euler, get_transform_from_rotation_translation, inverse_transform, get_rotation_translation_from_transform

# from dataset.nclt_visual_loop_dataset import EvaluationTuple, EvaluationSet, get_image_loader
from dataset.oxford_visual_loop_dataset import EvaluationTuple, EvaluationSet, get_image_loader

class Evaluator:
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float] = [1.5, 5, 20], k: int = 50, n_samples: int =None, debug: bool = False):
        # radius: list of thresholds (in meters) to consider an element from the map sequence a true positive
        # k: maximum number of nearest neighbours to consider
        # n_samples: number of samples taken from a query sequence (None=all query elements)

        assert os.path.exists(dataset_root), f"Cannot access dataset root: {dataset_root}"
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.eval_set_filepath = os.path.join(dataset_root, eval_set_pickle)
        self.device = device
        self.radius = radius
        self.k = k
        self.n_samples = n_samples
        self.debug = debug
        self.params = params

        assert os.path.exists(self.eval_set_filepath), f'Cannot access evaluation set pickle: {self.eval_set_filepath}'
        self.eval_set = EvaluationSet()
        self.eval_set.load(self.eval_set_filepath)
        if debug:
            # Make the same map set and query set in debug mdoe
            self.eval_set.map_set = self.eval_set.map_set[:4]
            self.eval_set.query_set = self.eval_set.map_set[:4]

        if n_samples is None or len(self.eval_set.query_set) <= n_samples:
            self.n_samples = len(self.eval_set.query_set)
        else:
            self.n_samples = n_samples

        self.pcim_loader = get_image_loader(self.dataset_type)

        # camera parameters
        self.transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        if self.dataset_type == 'nclt':
            image_meta_path = '/map-hl/gary.huang/public_dataset/NCLT/cam_params/image_meta.pkl'
        elif self.dataset_type == 'oxford':
            image_meta_path = '/map-hl/gary.huang/public_dataset/Oxford/dataset/image_meta.pkl'
        self.image_meta = load_pkl(image_meta_path)
        self.K = np.array(self.image_meta['K'])
        self.P = self.image_meta['T']
        if self.transform is not None:
            self.P = np.array([self.extrinsics_transform(e) for e in self.P])
        else:
            self.P = np.array(self.P)

        # image transform
        t = [transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.image_transform = transforms.Compose(t)

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

    def evaluate(self, model, *args, **kwargs):
        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()

        # Dictionary to store the number of true positives for different radius and NN number
        tp = {r: [0] * self.k for r in self.radius}
        query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        # Randomly sample n_samples clouds from the query sequence and NN search in the target sequence
        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Euclidean distance between the query and nn
            delta = query_pos - map_positions[nn_ndx]  # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)  # (k,) array
            # Count true positives for different radius and NN number
            tp = {r: [tp[r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in
                  self.radius}

        recall = {r: [tp[r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        # percentage of 'positive' queries (with at least one match in the map sequence within given radius)
        return {'recall': recall}

    def compute_embedding(self, pc, depth, imgs, model, *args, **kwargs):
        # This method must be implemented in inheriting classes
        # Must return embedding as a numpy vector
        raise NotImplementedError('Not implemented')

    def model2eval(self, model):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        model.eval()

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval(model)

        embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            assert os.path.exists(scan_filepath)
            pc, imgs = self.pcim_loader(scan_filepath)
            pc = torch.tensor(pc)

            embedding = self.compute_embedding(pc, imgs, model)
            if embeddings is None:
                embeddings = np.zeros((len(eval_subset), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[ndx] = embedding

        return embeddings

class GLEvaluator(Evaluator):
    # Evaluation of EgoNN methods on Mulan or Apollo SouthBay dataset
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params=None,
                 radius: List[float] = [2, 5, 10, 20], k: int = 20, n_samples=None, debug: bool = False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, params, radius, k, n_samples, debug=debug)
        self.cfg = params
        self.model = None
        # self.with_pose = self.cfg.model.with_local_feats
        self.with_pose  = True
        self.sample_limit = -1
        self.rerank_num = 20
        self.loop_thresh = 1.0

        self.extrinsics_dir = os.path.join('/home/g.huang/visual_loop_closure/dataset/Oxford', 'extrinsics')

    def save_sample(self, eval_subset, ndx, prefix='img'):
        e = eval_subset[ndx]
        if self.dataset_type == 'nclt':
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            query_imgs = self.pcim_loader(scan_filepath)
        elif self.dataset_type == 'oxford':
            query_imgs = self.pcim_loader(e.filepaths, False, self.extrinsics_dir)
        for i in range(len(query_imgs)):
            cv2.imwrite(prefix + '_' + str(i) + '.png', query_imgs[i][:, :, [2, 1, 0]])
        return

    def run(self, start=0, end=-1, step=1, write=True):
        map_positions = self.eval_set.get_map_positions() 
        query_positions = self.eval_set.get_query_positions()
        map_poses = self.eval_set.get_map_poses()
        query_poses = self.eval_set.get_query_poses()
        # import pdb; pdb.set_trace()
        map_loop_desc, map_pose_desc = self.compute_embeddings(self.eval_set.map_set)

        query_indexes = list(range(len(self.eval_set.query_set)))
        self.n_samples = len(self.eval_set.query_set)

        map_loop_desc = np.array(map_loop_desc)
        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {'tp': {r: [0] * self.k for r in self.radius}}
        global_metrics_rerank = {'tp': {r: [0] * self.k for r in self.radius}}
        metrics = {}
        non_static_sample = 0
        ryes, rtes, costs = [], [], []
        
        if write:
            f = open('eval_log_' + str(start) + '_' + str(end) + '_' + str(step) + '.txt', 'a')

        for query_ndx in tqdm.tqdm(query_indexes[start:][::step]):
             # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]
            query_pose = query_poses[query_ndx]
            # if np.linalg.norm(query_pos - map_positions[:self.sample_limit, ...], axis=1).min() > 2.:
            #     non_static_sample += 1
            #     continue

            # get query desciriptors
            with torch.no_grad():
                e = self.eval_set.query_set[query_ndx]
                if self.dataset_type == 'nclt':
                    scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
                    query_imgs = self.pcim_loader(scan_filepath)
                elif self.dataset_type == 'oxford':
                    query_imgs = self.pcim_loader(e.filepaths, False, self.extrinsics_dir)
                query_imgs = [self.image_transform(e) for e in query_imgs]
                query_imgs = np.stack(query_imgs)

                # prepare data dict
                qry_data_dict = {'query_img':  torch.from_numpy(query_imgs).unsqueeze(0).float(),
                                                'query_P':  torch.from_numpy(self.P).unsqueeze(0).float(), 
                                                'query_K':  torch.from_numpy(self.K).unsqueeze(0).float()}
                out_data_dict = self.model(to_cuda(qry_data_dict))
                query_embedding  = out_data_dict['loop_desc'].squeeze().detach().cpu().numpy()
                qry_pose_embedding  = out_data_dict['bev'].squeeze().detach().cpu().numpy()

            # Nearest neighbour search in the embedding space
            embed_dist = np.linalg.norm(map_loop_desc - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Metrics before re-rank
            delta = query_pos - map_positions[nn_ndx]       # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array
            global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}

            if write:
                for i in range(euclid_dist.shape[0]):
                    f.write(str(euclid_dist[i]) + ' ')
                f.write('\n')
                for i in range(nn_ndx.shape[0]):
                    f.write(str(nn_ndx[i]) + ' ')
                f.write('\n')

            # Re-ranking
            with torch.no_grad():
                ryes_rerank, rtes_rerank, costs_rerank = [], [], []
                rerank_pair_num = 0
                self.model.localizer.searching = False
                for i in range(self.rerank_num):
                    if embed_dist[nn_ndx[i]] > self.loop_thresh:
                        continue
                    else:
                        rerank_pair_num += 1

                    map_pose = map_poses[nn_ndx[i]]
                    map_pose_embedding = map_pose_desc[nn_ndx[i]]
                    pose_data_dict =  {'pose_desc_1': torch.from_numpy(qry_pose_embedding).float().unsqueeze(0), 'pose_desc_2': torch.from_numpy(map_pose_embedding).float().unsqueeze(0)}
                    pose_out_dict = self.model(to_cuda(pose_data_dict), localizing=True)
                    rel_yaw = pose_out_dict['rel_yaw']
                    rel_t = pose_out_dict['rel_t'][0][0].numpy()
                    cost = pose_out_dict['min_costs']

                    # calculate errors
                    rel_T = np.matmul(inverse_transform(query_pose), map_pose)
                    rel_R, rel_translation = get_rotation_translation_from_transform(rel_T)
                    rel_euler = rotation_to_euler(rel_R)
                    gt_yaw = rel_euler[2]
                    gt_t = rel_translation[:2]
                    diff = abs(rel_yaw - gt_yaw)
                    rye = min(360 - diff, diff)
                    rte = np.linalg.norm(gt_t - rel_t)

                    # save errors
                    ryes_rerank.append(rye)
                    rtes_rerank.append(rte)
                    costs_rerank.append(cost)

                sorted_indices = sorted(range(rerank_pair_num), key=lambda x: costs_rerank[x])
                nn_ndx_rerank = nn_ndx.copy()
                nn_ndx_rerank[:rerank_pair_num] = nn_ndx[:rerank_pair_num][sorted_indices]

                # estimate pose for the top-1 
                with torch.no_grad():
                    self.model.localizer.searching = True
                    map_pose = map_poses[nn_ndx_rerank[0]]
                    map_pose_embedding = map_pose_desc[nn_ndx_rerank[0]]
                    pose_data_dict =  {'pose_desc_1': torch.from_numpy(qry_pose_embedding).float().unsqueeze(0), 'pose_desc_2': torch.from_numpy(map_pose_embedding).float().unsqueeze(0)}

                    pose_out_dict = self.model(to_cuda(pose_data_dict), localizing=True)
                    rel_yaw = pose_out_dict['rel_yaw']
                    rel_t = pose_out_dict['rel_t'][0][0].numpy()
                    cost = pose_out_dict['min_costs']

                    # calculate errors
                    rel_T = np.matmul(inverse_transform(query_pose), map_pose)
                    rel_R, rel_translation = get_rotation_translation_from_transform(rel_T)
                    rel_euler = rotation_to_euler(rel_R)
                    gt_yaw = rel_euler[2]
                    gt_t = rel_translation[:2]
                    diff = abs(rel_yaw - gt_yaw)
                    rye = min(360 - diff, diff)
                    rte = np.linalg.norm(gt_t - rel_t)

                    # save errors
                    ryes.append(rye) 
                    rtes.append(rte)
                    costs.append(cost)

                # GLOBAL DESCRIPTOR EVALUATION
                # Euclidean distance between the query and nn
                # Here we use non-icp refined poses, but for the global descriptor it's fine
                delta_rerank = query_pos - map_positions[nn_ndx_rerank]       # (k, 2) array
                euclid_dist_rerank = np.linalg.norm(delta_rerank, axis=1)     # (k,) array
                global_metrics_rerank['tp'] = {r: [global_metrics_rerank['tp'][r][nn] + (1 if (euclid_dist_rerank[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}

                if write:
                    for i in range(euclid_dist_rerank.shape[0]):
                        f.write(str(euclid_dist_rerank[i]) + ' ')
                    f.write('\n')
                    for i in range(nn_ndx_rerank.shape[0]):
                        f.write(str(nn_ndx_rerank[i]) + ' ')
                    f.write('\n')

                    f.write(str(rye) + ' ' + str(rte) + ' ' + str(cost))
                    print(rye, rte, cost)
                    f.write('\n')

                # if abs(gt_yaw) > 30 and abs(rye) < 1 and abs(rte) < 0.5:
                #     self.save_sample(self.eval_set.query_set, query_ndx, prefix='query')
                #     self.save_sample(self.eval_set.map_set, nn_ndx_rerank[0], prefix='map')
                #     pdb.set_trace()

        # Calculate mean metrics
        global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / (self.n_samples - non_static_sample) for nn in range(self.k)] for r in self.radius}
        metrics['Recall@1'] = global_metrics['recall'][2][0]
        metrics['Recall@5'] = global_metrics['recall'][2][4]
        metrics['Recall@10'] = global_metrics['recall'][2][9]
        metrics['Recall@20'] = global_metrics['recall'][2][19]

        global_metrics_rerank["recall"] = {r: [global_metrics_rerank['tp'][r][nn] / (self.n_samples - non_static_sample) for nn in range(self.k)] for r in self.radius}
        metrics['Recall_rerank@1'] = global_metrics_rerank['recall'][2][0]
        metrics['Recall_rerank@5'] = global_metrics_rerank['recall'][2][4]
        metrics['Recall_rerank@10'] = global_metrics_rerank['recall'][2][9]
        metrics['Recall_rerank@20'] = global_metrics_rerank['recall'][2][19]

        ryes = np.array(ryes)
        rtes = np.array(rtes)
        costs = np.array(costs)
        metrics['mRRE'] = ryes.mean()
        metrics['mRTE'] = rtes.mean()
        return metrics
            
    def run_x(self):
        if self.with_pose:
            map_loop_desc, map_pose_desc = self.compute_embeddings(self.eval_set.map_set)
            qry_loop_desc, qry_pose_desc = self.compute_embeddings(self.eval_set.query_set)
        else:
            map_loop_desc = self.compute_embeddings(self.eval_set.map_set)
            qry_loop_desc = self.compute_embeddings(self.eval_set.query_set)        

        map_positions = self.eval_set.get_map_positions() #(M, 2)
        query_positions = self.eval_set.get_query_positions() # (N, 2)
        map_poses = self.eval_set.get_map_poses() #(M, 4, 4)
        query_poses = self.eval_set.get_query_poses() # (N, 4, 4)

        if self.n_samples is None or len(qry_loop_desc) <= self.n_samples:
            query_indexes = list(range(len(qry_loop_desc)))
            self.n_samples = len(qry_loop_desc)
        else:
            query_indexes = random.sample(range(len(qry_loop_desc)), self.n_samples)

        qry_loop_desc = np.array(qry_loop_desc)
        map_loop_desc = np.array(map_loop_desc)

        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {'tp': {r: [0] * self.k for r in self.radius}}
        global_metrics_rerank = {'tp': {r: [0] * self.k for r in self.radius}}
        metrics = {}
        non_static_sample = 0
        ryes, rtes, costs = [], [], []

        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]
            query_pose = query_poses[query_ndx]
            if np.linalg.norm(query_pos - map_positions[:self.sample_limit, ...], axis=1).min() > 2.:
                non_static_sample += 1
                continue

            # Nearest neighbour search in the embedding space
            query_embedding = qry_loop_desc[query_ndx]
            embed_dist = np.linalg.norm(map_loop_desc - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Metrics before re-rank
            delta = query_pos - map_positions[nn_ndx]       # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array
            global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}

            # Re-ranking
            if self.with_pose:
                with torch.no_grad():
                    qry_pose_embedding = qry_pose_desc[query_ndx]
                    ryes_rerank, rtes_rerank, costs_rerank = [], [], []
                    rerank_pair_num = 0
                    for i in range(self.rerank_num):
                        if embed_dist[nn_ndx[i]] > self.loop_thresh:
                            continue
                        else:
                            rerank_pair_num += 1

                        map_pose = map_poses[nn_ndx[i]]
                        map_pose_embedding = map_pose_desc[nn_ndx[i]]
                        pose_data_dict =  {'pose_desc_1': torch.from_numpy(qry_pose_embedding).float().unsqueeze(0), 'pose_desc_2': torch.from_numpy(map_pose_embedding).float().unsqueeze(0)}
                        pose_out_dict = self.model(to_cuda(pose_data_dict), localizing=True)
                        rel_yaw = pose_out_dict['rel_yaw']
                        rel_t = pose_out_dict['rel_t'][0][0].numpy()
                        cost = pose_out_dict['min_costs']

                        # calculate errors
                        rel_T = np.matmul(inverse_transform(query_pose), map_pose)
                        rel_R, rel_translation = get_rotation_translation_from_transform(rel_T)
                        rel_euler = rotation_to_euler(rel_R)
                        gt_yaw = rel_euler[2]
                        gt_t = rel_translation[:2]
                        diff = abs(rel_yaw - gt_yaw)
                        rye = min(360 - diff, diff)
                        rte = np.linalg.norm(gt_t - rel_t)

                        # save errors
                        ryes_rerank.append(rye)
                        rtes_rerank.append(rte)
                        costs_rerank.append(cost)

                sorted_indices = sorted(range(rerank_pair_num), key=lambda x: costs_rerank[x])
                ryes.append(ryes_rerank[sorted_indices[0]])
                rtes.append(rtes_rerank[sorted_indices[0]])
                costs.append(costs_rerank[sorted_indices[0]])
                nn_ndx_rerank = nn_ndx.copy()
                nn_ndx_rerank[:rerank_pair_num] = nn_ndx[:rerank_pair_num][sorted_indices]

                # GLOBAL DESCRIPTOR EVALUATION
                # Euclidean distance between the query and nn
                # Here we use non-icp refined poses, but for the global descriptor it's fine
                delta_rerank = query_pos - map_positions[nn_ndx_rerank]       # (k, 2) array
                euclid_dist_rerank = np.linalg.norm(delta_rerank, axis=1)     # (k,) array
                global_metrics_rerank['tp'] = {r: [global_metrics_rerank['tp'][r][nn] + (1 if (euclid_dist_rerank[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            # debug
            # if euclid_dist[0] > 10:
            #     self.save_sample(self.eval_set.query_set, query_ndx, prefix='query')
            #     self.save_sample(self.eval_set.map_set, nn_ndx[0], prefix='map')
            #     pdb.set_trace()

        # Calculate mean metrics
        global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / (self.n_samples - non_static_sample) for nn in range(self.k)] for r in self.radius}
        metrics['Recall@1'] = global_metrics['recall'][2][0]
        metrics['Recall@5'] = global_metrics['recall'][2][4]
        metrics['Recall@10'] = global_metrics['recall'][2][9]
        metrics['Recall@20'] = global_metrics['recall'][2][19]

        if self.with_pose:
            global_metrics_rerank["recall"] = {r: [global_metrics_rerank['tp'][r][nn] / (self.n_samples - non_static_sample) for nn in range(self.k)] for r in self.radius}
            metrics['Recall_rerank@1'] = global_metrics_rerank['recall'][2][0]
            metrics['Recall_rerank@5'] = global_metrics_rerank['recall'][2][4]
            metrics['Recall_rerank@10'] = global_metrics_rerank['recall'][2][9]
            metrics['Recall_rerank@20'] = global_metrics_rerank['recall'][2][19]

        if self.with_pose:
            ryes = np.array(ryes)
            rtes = np.array(rtes)
            costs = np.array(costs)
            metrics['mRRE'] = ryes.mean()
            metrics['mRTE'] = rtes.mean()
        return metrics

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], batch_size=8):
        loop_desc_list, pose_desc_list = [], []
        position_list, pose_list = [], []
        with torch.no_grad():
            for ndx, e in tqdm.tqdm(enumerate(eval_subset[:self.sample_limit])):
            # for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
                if self.dataset_type == 'nclt':
                    scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
                    query_imgs = self.pcim_loader(scan_filepath)
                elif self.dataset_type == 'oxford':
                    query_imgs = self.pcim_loader(e.filepaths, False, self.extrinsics_dir)

                query_imgs = [self.image_transform(e) for e in query_imgs]
                if ndx % batch_size == 0:
                    query_imgs_batch = np.stack(query_imgs)[np.newaxis, ...]
                else:
                    query_imgs_batch = np.concatenate([query_imgs_batch, np.stack(query_imgs)[np.newaxis, ...]])

                # prepare data dict
                if ndx % batch_size == (batch_size - 1) or ndx == (len(eval_subset) - 1):
                    qry_data_dict = {'query_img':  torch.from_numpy(query_imgs_batch).float(),
                                                    'query_P':  torch.from_numpy(self.P).unsqueeze(0).repeat(query_imgs_batch.shape[0], 1, 1, 1).float(), 
                                                    'query_K':  torch.from_numpy(self.K).unsqueeze(0).repeat(query_imgs_batch.shape[0], 1, 1, 1).float()}
                                        
                    out_data_dict = self.model(to_cuda(qry_data_dict))
                    loop_desc = release_cuda(out_data_dict['loop_desc'])
                    for i in range(loop_desc.shape[0]):
                        loop_desc_list.append(loop_desc[i, ...])
                    if self.with_pose:
                        pose_desc = release_cuda(out_data_dict['bev'])
                        for i in range(pose_desc.shape[0]):
                            pose_desc_list.append(pose_desc[i, ...])
        
        if self.with_pose:
            return loop_desc_list, pose_desc_list
        else:
            return loop_desc_list
