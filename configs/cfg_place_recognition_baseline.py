import argparse
import os
import os.path as osp
import math
import numpy as np

from easydict import EasyDict as edict
from utils.common import ensure_dir

_C = edict()
# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = 'place_recognition_baseline'
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)

# basic
_C.seed = 1234

# dataset
_C.data = edict()
_C.data.dataset = 'nio' #'nclt'
_C.data.dataset_path = '/map-hl/gary.huang/Visual_Loop_Closure/training/'
_C.data.loop_pkl_path_train = './dataset/loop_closure.pkl'#'./dataset/loop_closure_train_sample.pkl'# 
_C.data.loop_pkl_path_eval = './dataset/loop_closure_eval.pkl'
_C.data.with_amb = False
_C.data.img_width = 448
_C.data.img_height = 256
_C.data.dataset_length = -1
_C.data.neg_num = 4

# training
_C.train = edict()
_C.train.batch_size = 2
_C.train.num_workers = 12

# evaling
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 1

# optimizer
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.90
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-2
_C.optim.max_epoch = 40
_C.optim.grad_acc_steps = 1

# loss function
_C.loss = edict()
_C.loss.margin_triplet = 0.1
_C.loss.margin_pos = 0.1
_C.loss.margin_neg = 0.2

# model
_C.model = edict()
_C.model.backbone = 'res101'
_C.model.output_dim = 256
_C.model.ffn_dim = 1028
_C.model.feature_dim = 256
_C.model.num_layers = 6
_C.model.desc_dim = 1024
_C.model.with_pos_emb=True
_C.model.with_local_feats = False

_C.model.scene_centroid = 0.0, 0.0, 0.0
_C.model.xbounds = (-15, 15) #(-70, 70)
_C.model.ybounds = (-5, 0) #(-1.5, 0) 
_C.model.zbounds = (-15, 15)
_C.model.X = 100
_C.model.Y = 10 # 20
_C.model.Z = 100
_C.model.radius = 40
_C.model.theta = 120
_C.model.normalize = True
_C.model.grid_size = 0.3

# localizer
_C.localizer = edict()
_C.localizer.pose_estimate = 'polar' # regress, match
_C.localizer.loop_detect = 'disco' # distance, score 
_C.localizer.loop_interact = False
_C.localizer.loop_thresh = 0.1
_C.localizer.pos_radius = 0.3 # for matching
_C.localizer.distance_threshold = 0.3 # for target generating
_C.localizer.num_targets = 256
_C.localizer.num_correspondences = 512
_C.localizer.dual_normalization = True

# online inferernce
_C.infer = edict()
_C.infer.dual_reg = False
_C.infer.loop_thresh = 0.1
_C.infer.match_thresh = 0.745
_C.infer.reg_thresh = 20000
_C.infer.delta_z_thresh = 2.
_C.infer.p_thresh = 3.
_C.infer.r_thresh = 3.

def make_cfg():
    return _C

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args

def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')

if __name__ == '__main__':
    main()
