import pdb
import os
import cv2
import torch
import numpy as np

from tqdm import tqdm
from scipy.spatial.distance import cdist
from eval.matching import best_match_per_query, thresholding
from eval.metrics import createPR, recallAt100precision, recallAtK

from utils.registration import compute_registration_error
from utils.torch_utils import to_cuda, all_reduce_tensors, release_cuda, initialize
from utils.geometry import quaternion_to_rotation, rotation_to_quaternion, rotation_to_euler, get_transform_from_rotation_translation, inverse_transform, get_rotation_translation_from_transform

import pickle

def compute_l2_distance_matrix(desc1, desc2):
    desc1_array = np.array(desc1)
    desc2_array = np.array(desc2)
    distance_matrix = cdist(desc1_array, desc2_array, metric='euclidean')
    return distance_matrix

def compute_position_distance_matrix(pos1, pos2):
    desc1_array = np.array(pos1)
    desc2_array = np.array(pos2)
    xy_matrix = cdist(desc1_array[:,0:2], desc2_array[:,0:2], metric='euclidean')
    z_matrix = cdist(desc1_array[:,2:], desc2_array[:,2:], metric='euclidean')
    diff_floor = np.where(z_matrix > 4.)
    z_matrix[diff_floor] = 100.
    distance_matrix = z_matrix + xy_matrix
    return distance_matrix, np.concatenate([xy_matrix[:, :, np.newaxis], z_matrix[:, :, np.newaxis]], 2)

def remove_diagonal_elements(indices, k, s):
    idx, idy = indices
    filtered_indices = np.array([(i, j) for i, j in zip(idx, idy) if abs(i - j) >= k // s])
    out_indices = (filtered_indices[:, 0], filtered_indices[:, 1])
    return out_indices

class LCD_evaluator():
    def __init__(self, cfg, dataset,  model,  
                            loop_thresh = 0.1, 
                            positive_thresh=10., negative_min_thresh=20.,):
            self.dataset = dataset
            self.model = model

            self.positive_thresh = positive_thresh
            self.negative_min_thresh = negative_min_thresh
            self.loop_thresh = loop_thresh

            # self.with_pose = cfg.model.with_local_feats
            self.with_pose = False
            self.conf = 0.01
            self.cost_thresh = 0.775
            self.pose_thresh = 0.745
            self.k = 1
            self.s = 3

    def write_similarity(self, D, name):
        cv2.imwrite(name + '.png', D * 255 / np.max(D))
        return

    def save_distance(self, D, name):
        np.save(name + '.npy', D)
        return
    
    def loop_closure_detection(self, descs, positions, key=''):
        # calculate recall and precision of loop detection results under threshold 0.1
        S = compute_l2_distance_matrix(descs, descs) #(N, N)
        D, D_real = compute_position_distance_matrix(positions, positions) #(N, N)
        self.write_similarity(S, './result/sim_' + key)
        self.write_similarity(D, './result/dis_' + key)
        self.save_distance(S, './result/sim_' + key)
        self.save_distance(D, './result/dis_' + key)
        self.save_distance(D_real, './result/dis_real_' + key)
        pos_thresh = self.loop_thresh
        k, s = self.k, self.s

        # sample for keyframes
        S = S[::s, ::s]
        D = D[::s, ::s]
        D_real = D_real[::s, ::s, :]

        #1 . correct detect loop/ whole detected loop (under thresh) and extract wrong-detected sample
        pos_sim = remove_diagonal_elements(np.where(S < pos_thresh), k, s)
        neg_sim = remove_diagonal_elements(np.where(S > pos_thresh), k, s)
        dis_pos, dis_neg = D_real[:, :, 0][pos_sim], D_real[:, :, 0][neg_sim]
        dis_height_pos, dis_height_neg = D_real[:, :, 1][pos_sim], D_real[:, :, 1][neg_sim]
        precision_2_a = np.where((dis_pos < 2) & (dis_height_pos < 3.))[0].shape[0] / pos_sim[0].shape[0]
        precision_5_a = np.where((dis_pos < 5) & (dis_height_pos < 3.))[0].shape[0] / pos_sim[0].shape[0]
        precision_10_a = np.where((dis_pos < 10) & (dis_height_pos < 3.))[0].shape[0] / pos_sim[0].shape[0]
        precision_20_a = np.where((dis_pos < 20) & (dis_height_pos < 3.))[0].shape[0] / pos_sim[0].shape[0]

        pos_sim = remove_diagonal_elements(np.where((S < pos_thresh) & (D_real[:, :, 0] < 20)), k, s)
        dis_pos, dis_neg = D_real[:, :, 0][pos_sim], D_real[:, :, 0][neg_sim]    
        dis_height_pos, dis_height_neg = D_real[:, :, 1][pos_sim], D_real[:, :, 1][neg_sim]
        precision_2_20 = np.where((dis_pos < 2) & (dis_height_pos < 3.))[0].shape[0] / pos_sim[0].shape[0]
        precision_5_20 = np.where((dis_pos < 5) & (dis_height_pos < 3.))[0].shape[0] / pos_sim[0].shape[0]
        precision_10_20 = np.where((dis_pos < 10) & (dis_height_pos < 3.))[0].shape[0] / pos_sim[0].shape[0]

        # 2. wrong detect loop (especially different layer)/ whole detected loop and extract wrong-detected sample
        pos_sim = remove_diagonal_elements(np.where(S < pos_thresh), k, s)
        neg_sim = remove_diagonal_elements(np.where(S > pos_thresh), k, s)
        dis_pos, dis_neg = D_real[:, :, 0][pos_sim], D_real[:, :, 0][neg_sim]
        dis_height_pos, dis_height_neg = D_real[:, :, 1][pos_sim], D_real[:, :, 1][neg_sim]
        error_2_a = np.where((dis_pos < 2) & (dis_height_pos > 3.))[0].shape[0] / pos_sim[0].shape[0]
        error_5_a = np.where((dis_pos < 5) & (dis_height_pos > 3.))[0].shape[0] / pos_sim[0].shape[0]
        error_10_a = np.where((dis_pos < 10) & (dis_height_pos > 3.))[0].shape[0] / pos_sim[0].shape[0]

        pos_sim = remove_diagonal_elements(np.where((S < pos_thresh) & (D_real[:, :, 0] < 20)), k, s)
        dis_pos, dis_neg = D_real[:, :, 0][pos_sim], D_real[:, :, 0][neg_sim]    
        dis_height_pos, dis_height_neg = D_real[:, :, 1][pos_sim], D_real[:, :, 1][neg_sim]
        error_2_20 = np.where((dis_pos < 2) & (dis_height_pos > 3.))[0].shape[0] / pos_sim[0].shape[0]
        error_5_20 = np.where((dis_pos < 5) & (dis_height_pos > 3.))[0].shape[0] / pos_sim[0].shape[0]
        error_10_20 = np.where((dis_pos < 10) & (dis_height_pos > 3.))[0].shape[0] / pos_sim[0].shape[0]

        # 3. correctly detected loop / whole loop (under thresh) and extract un-detected sample
        pos_dis = remove_diagonal_elements(np.where((D_real[:, :, 0] < 2.) & (D_real[:, :, 1] < 3.)), k, s)
        sim_pos = S[pos_dis]
        recall_2 = np.where(sim_pos < pos_thresh)[0].shape[0] / pos_dis[0].shape[0]
        pos_dis = remove_diagonal_elements(np.where((D_real[:, :, 0] < 5.) & (D_real[:, :, 1] < 3.)), k, s)
        sim_pos = S[pos_dis]
        recall_5 = np.where(sim_pos < pos_thresh)[0].shape[0] / pos_dis[0].shape[0]
        pos_dis =  remove_diagonal_elements(np.where((D_real[:, :, 0] < 10.) & (D_real[:, :, 1] < 3.)), k, s)
        sim_pos = S[pos_dis]
        recall_10 = np.where(sim_pos < pos_thresh)[0].shape[0] / pos_dis[0].shape[0]
        return recall_2, recall_5, recall_10, \
                      precision_2_a, precision_5_a, precision_10_a, \
                      precision_2_20, precision_5_20, precision_10_20, \
                      error_2_a, error_5_a, error_10_a, \
                      error_2_20, error_5_20, error_10_20, \
                      S, D_real

    def _relative_pose_estimation_regress_six_dof(self, pose_desc_1, pose_desc_2, pose_T_1, pose_T_2, batch_size=8):
        M = pose_desc_1.shape[0]
        for i in range(0, M, batch_size):
            batch_pose_desc_1 = to_cuda(torch.from_numpy(pose_desc_1[i:i+batch_size]))
            batch_pose_desc_2 = to_cuda(torch.from_numpy(pose_desc_2[i:i+batch_size]))
            input_dict = {'pose_desc_1': batch_pose_desc_1, 'pose_desc_2': batch_pose_desc_2}
            with torch.no_grad():
                out_dict = self.model(to_cuda(input_dict), localizing=True)
                if i == 0:
                    rel_t = release_cuda(out_dict['rel_t'])
                    rel_q = release_cuda(out_dict['rel_q'])
                else:
                    rel_t = np.concatenate([rel_t, release_cuda(out_dict['rel_t'])], axis=0) # (M, 3)
                    rel_q = np.concatenate([rel_q, release_cuda(out_dict['rel_q'])], axis=0) # (M, 4)

        mRRE, mRTE = 0., 0.
        for i in range(M):
            pred_R, pred_t = quaternion_to_rotation(rel_q[i]), rel_t[i]
            rel_T_pred = np.identity(4, dtype=np.float32)
            rel_T_pred[0:3, 0:3] = pred_R
            rel_T_pred[0:3, 3] = pred_t
            rel_T_gt =  np.matmul(inverse_transform(pose_T_1[i,...]), pose_T_2[i,...])

            rre, rte = compute_registration_error(rel_T_pred, rel_T_gt)
            mRRE += rre
            mRTE += rte
        
        mRRE, mRTE = mRRE / M, mRTE / M
        return mRRE, mRTE

    def _relative_pose_estimation_regress_three_dof(self, pose_desc_1, pose_desc_2, pose_T_1, pose_T_2, tp='det', batch_size=12):
        out_list = []
        M = pose_desc_1.shape[0]
        for i in range(0, M, batch_size):
            batch_pose_desc_1 = to_cuda(torch.from_numpy(pose_desc_1[i:i+batch_size]))
            batch_pose_desc_2 = to_cuda(torch.from_numpy(pose_desc_2[i:i+batch_size]))
            input_dict = {'pose_desc_1': batch_pose_desc_1, 'pose_desc_2': batch_pose_desc_2}
            with torch.no_grad():
                out_dict = self.model(to_cuda(input_dict), localizing=True)
                if i == 0:
                    rel_t = release_cuda(out_dict['rel_t'])
                    rel_yaw = release_cuda(out_dict['rel_yaw'])
                else:
                    rel_t = np.concatenate([rel_t, release_cuda(out_dict['rel_t'])], axis=0) # (M, 2)
                    rel_yaw = np.concatenate([rel_yaw, release_cuda(out_dict['rel_yaw'])], axis=0) # (M, 1)

        mRRE, mRTE = 0., 0.
        for i in range(M):
            rel_R_gt, rel_t_gt =  get_rotation_translation_from_transform(np.matmul(inverse_transform(pose_T_1[i,...]), pose_T_2[i,...]))
            rel_euler_gt = rotation_to_euler(rel_R_gt)
            rre = min((360 - abs(rel_euler_gt[2] - rel_yaw[i])), abs(rel_euler_gt[2] - rel_yaw[i]))
            rte = np.linalg.norm(rel_t_gt[:2] - rel_t[i]).mean()
            mRRE += rre
            mRTE += rte
            out_list.append([rel_euler_gt[2], rel_t_gt[0], rel_t_gt[1], rel_yaw[i]])
        
        with open('./result/' + self.name + '_' + tp + '_detailed.pkl', 'wb') as file:
            pickle.dump(out_list, file)

        mRRE, mRTE = mRRE / M, mRTE / M
        return mRRE, mRTE

    def relative_pose_estimation_regress(self, S, D, pose_descs, poses, pair_limit=360):
        pose_descs = np.array(pose_descs) # (N, C)
        poses = np.array(poses) # (N, 4, 4)

        # compute mrre and mrte of the corretly detected loop pairs
        positive_detected_loops = np.where((S < self.loop_thresh) & (D < 5.))
        indices = np.random.choice(positive_detected_loops[0].shape[0], size=pair_limit, replace=False)
        pose_desc_1 = pose_descs[positive_detected_loops[0][indices], :] # (M, C)
        pose_desc_2 = pose_descs[positive_detected_loops[1][indices], :] # (M, C)
        pose_T_1 = poses[positive_detected_loops[0], ...] # (M, 4, 4)
        pose_T_2 = poses[positive_detected_loops[1], ...] # (M, 4, 4)
        # mRRE_det, mRTE_det = self._relative_pose_estimation_regress_six_dof(pose_desc_1, pose_desc_2, pose_T_1, pose_T_2)
        mRRE_det, mRTE_det = self._relative_pose_estimation_regress_three_dof(pose_desc_1, pose_desc_2, pose_T_1, pose_T_2, 'det')

        # compute mrre and mrte of the whole ground-truth loops
        all_loops = np.where(D < 5.)
        indices = np.random.choice(all_loops[0].shape[0], size=pair_limit, replace=False)
        pose_desc_1 = pose_descs[all_loops[0][indices], :] # (M, C)
        pose_desc_2 = pose_descs[all_loops[1][indices], :] # (M, C)
        pose_T_1 = poses[all_loops[0], ...] # (M, 4, 4)
        pose_T_2 = poses[all_loops[1], ...] # (M, 4, 4)
        # mRRE_all, mRTE_all = self._relative_pose_estimation_regress_six_dof(pose_desc_1, pose_desc_2, pose_T_1, pose_T_2)
        mRRE_all, mRTE_all = self._relative_pose_estimation_regress_three_dof(pose_desc_1, pose_desc_2, pose_T_1, pose_T_2, 'all')

        return mRRE_det, mRTE_det, mRRE_all, mRTE_all

    def run(self, place_dict, name='eval'):
        loop_desc_list, pose_desc_list, name_list  = [], [], []
        position_list, pose_list = [], []
        out_dict = {}
        self.name = name
        with torch.no_grad():
            for key in tqdm(list(place_dict.keys())):
                data_dict = self.dataset.load_meta_data_evaling(place_dict[key])
                qry_data_dict = {'query_img':  data_dict['query_imgs'], 'query_P':  data_dict['query_P'], 'query_K':  data_dict['query_K']}
                out_data_dict = self.model(to_cuda(qry_data_dict))
                loop_desc = out_data_dict['loop_desc'].squeeze()
                if self.with_pose:
                    pose_desc = out_data_dict['bev'].squeeze()
                    pose_desc_list.append(release_cuda(pose_desc))

                loop_desc_list.append(release_cuda(loop_desc))
                pose_list.append(release_cuda(data_dict['query_T'].squeeze()))
                position_list.append(release_cuda(data_dict['query_T'][0, 0:3, 3].squeeze()))
                name_list.append(key)
        
        recall_2, recall_5, recall_10, precision_2_a, precision_5_a, precision_10_a, precision_2_20, precision_5_20, precision_10_20, error_2_a, error_5_a, error_10_a, error_2_20, error_5_20, error_10_20, S, D_real = self.loop_closure_detection(loop_desc_list, position_list, name)
        out_dict['Recall_2'] = recall_2
        out_dict['Precision_10'] = precision_10_a
        print(out_dict)
        if self.with_pose:
            mRRE_all, mRTE_all, pos_to_neg_correct, pos_to_neg_wrong, loop_rate = self.relative_pose_estimation_match(S, D_real, pose_desc_list, pose_list)
            out_dict['mRRE_all'] = mRRE_all
            out_dict['mRTE_all'] = mRTE_all
            out_dict['loop_rate'] = loop_rate
            # re-calculate recall and precision
            pos_dis = np.where((D_real[:, :, 0] < 2.) & (D_real[:, :, 1] < 3.))
            sim_pos = S[pos_dis]
            recall_2_rerank = (np.where(sim_pos < self.loop_thresh)[0].shape[0] - pos_to_neg_wrong) / pos_dis[0].shape[0]
            pos_sim = np.where(S < self.loop_thresh)
            neg_sim = np.where(S > self.loop_thresh)
            dis_pos, dis_neg = D_real[:, :, 0][pos_sim], D_real[:, :, 0][neg_sim]
            dis_height_pos, dis_height_neg = D_real[:, :, 1][pos_sim], D_real[:, :, 1][neg_sim]
            precision_10_a_rerank = (np.where((dis_pos < 10) & (dis_height_pos < 3.))[0].shape[0] - pos_to_neg_wrong) / (pos_sim[0].shape[0] - pos_to_neg_wrong - pos_to_neg_correct)
            out_dict['Recall_2_rerank'] = recall_2_rerank
            out_dict['Precision_10_rerank'] = precision_10_a_rerank
            print(out_dict)
        return out_dict

    def _relative_pose_estimation_match(self, bevs, poses, loops, S, D, num_limit=-1):
        mRRE, mRTE = 0, 0
        pos_to_neg_correct, pos_to_neg_wrong = 0, 0
        out_list = []
        if num_limit != -1:
            N = num_limit
            indices = np.random.choice(loops[0].shape[0], size=num_limit, replace=False)
            loop_idqs, loop_idbs = loops[0][indices],  loops[1][indices]
        else:
            N = loops[0].shape[0]
            loop_idqs, loop_idbs = loops[0], loops[1]

        count = 0
        valid = 0
        with torch.no_grad():
            for idx in tqdm(range(N)):
                idq, idb = loop_idqs[idx], loop_idbs[idx]
                if idb - idq <= self.k // self.s:
                    continue
                count += 1
                pose_data_dict =  {'pose_desc_1': torch.from_numpy(bevs[idq]).float().unsqueeze(0), 'pose_desc_2': torch.from_numpy(bevs[idb]).float().unsqueeze(0)}
                
                # out result
                pose_out_dict = self.model(to_cuda(pose_data_dict), localizing=True)
                rel_yaw = pose_out_dict['rel_yaw']
                rel_t = pose_out_dict['rel_t'][0][0].numpy()
                cost = pose_out_dict['min_costs']
                
                # calcluate pose estimation error
                query_pose, base_pose = poses[idq], poses[idb]
                rel_T = np.matmul(inverse_transform(query_pose), base_pose)
                rel_R, rel_translation = get_rotation_translation_from_transform(rel_T)
                rel_euler = rotation_to_euler(rel_R)
                gt_yaw = rel_euler[2]
                gt_t = rel_translation[:2]
                diff = abs(rel_yaw - gt_yaw)
                rye = min(360 - diff, diff)
                rte = np.linalg.norm(gt_t - rel_t)

                # re-ranking
                loop_dis, xy_dis, h_dis = S[idq, idb], D[idq, idb, 0], D[idq, idb, 1]
                if loop_dis > (self.loop_thresh - self.conf) and cost > self.cost_thresh:
                    if xy_dis > 10. or h_dis > 3.:
                        pos_to_neg_correct += 1
                    elif xy_dis <2 and h_dis < 3:
                        pos_to_neg_wrong += 1

                if cost < self.pose_thresh:
                    valid += 1
                    mRRE += rye 
                    mRTE += rte

                # save results
                out_list.append([S[idq, idb], D[idq, idb, 0], D[idq, idb, 1], rel_yaw, rel_t[0], rel_t[1], gt_yaw, gt_t[0], gt_t[1], rye, rte, cost])
                with open('./result/' + self.name + '_detailed.pkl', 'wb') as file:
                    pickle.dump(out_list, file)

        return mRRE / (valid + 0.00001), mRTE / (valid + 0.00001), pos_to_neg_correct, pos_to_neg_wrong, valid / (count + 0.00001)

    def relative_pose_estimation_match(self, S, D, bevs, poses):
        poses = poses[::self.s] # (N, 4, 4)
        bevs = bevs[::self.s] # (N, 256, 100, 100)

        # # compute mrre and mrte of the corretly detected loop pairs
        # positive_detected_loops = np.where((S < self.loop_thresh) & (D < 5.))
        # mRRE_det, mRTE_det = self._relative_pose_estimation_match(place_dict, poses, positive_detected_loops)

        # # compute mrre and mrte of the whole ground-truth loops
        # all_loops = np.where(D < 5.)
        # mRRE_all, mRTE_all = self._relative_pose_estimation_match(place_dict, poses, all_loops)

        # compute mrre and mrte of the whole detected loop pairs
        detected_loops = np.where(S < self.loop_thresh)
        mRRE_all, mRTE_all, pos_to_neg_correct, pos_to_neg_wrong, loop_rate = self._relative_pose_estimation_match(bevs, poses, detected_loops, S, D)
        return mRRE_all, mRTE_all, pos_to_neg_correct, pos_to_neg_wrong, loop_rate

    def run_match(self, place_dict, name='eval'):
        loop_desc_list, pose_desc_list, name_list  = [], [], []
        position_list, pose_list = [], []
        out_dict = {}
        with torch.no_grad():
            for key in tqdm(list(place_dict.keys())):
                data_dict = self.dataset.load_meta_data_evaling(place_dict[key])
                qry_data_dict = {'query_img':  data_dict['query_imgs'], 'query_P':  data_dict['query_P'], 'query_K':  data_dict['query_K']}
                out_data_dict = self.model(to_cuda(qry_data_dict))
                loop_desc = out_data_dict['loop_desc'].squeeze()

                loop_desc_list.append(release_cuda(loop_desc))
                pose_list.append(release_cuda(data_dict['query_T'].squeeze()))
                position_list.append(release_cuda(data_dict['query_T'][0, 0:3, 3].squeeze()))
                name_list.append(key)
        
        recall, precision, S, D = self.loop_closure_detection(loop_desc_list, position_list, name)
        out_dict['Recall'] = recall
        out_dict['Precision'] = precision
        if self.with_pose:
            mRRE_all, mRTE_all = self.relative_pose_estimation_match(S, D, place_dict, pose_list)
            out_dict['mRRE_all'] = mRRE_all
            out_dict['mRTE_all'] = mRTE_all
        return out_dict