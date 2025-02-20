import os
import pdb
import time
import tqdm
import argparse
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from collections import OrderedDict

from utils.timer import Timer
from utils.logger import Logger
from utils.common import get_log_string
from utils.summary_board import SummaryBoard
from utils.torch_utils import to_cuda, all_reduce_tensors, release_cuda, initialize

from eval.evaluator import LCD_evaluator
from model.place_recognition_branch import create_model
from loss.loop_closure_loss import LossFunction_Total, compute_metrics

from dataset.collate import SimpleSingleCollateFnPackMode
from dataset.nio_visual_loop_dataset import NioVisualLoopDataset
from dataset.nclt_visual_loop_dataset import NCLTVisualLoopDataset
from dataset.oxford_visual_loop_dataset import OxfordVisualLoopDataset

parser = argparse.ArgumentParser(description='multi-camera visual loop closure')
parser.add_argument('--resume', default=True, action='store_true', help='resume training')
parser.add_argument('--pretrain', default=True, action='store_true', help='pretrain from resume')
parser.add_argument('--finetune', default=True, action='store_true', help='finetuning on pretrained model')
parser.add_argument('--debug', action='store_true', help='resume training')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--dataset', type=str, default='nio', choices=['nio', 'nclt', 'oxford'])
parser.add_argument('--local_rank', type=int, default=4, help='local rank for ddp')

class Trainer():
        def __init__(self, cfg, args,
                                  cudnn_deterministic=True, autograd_anomaly_detection=False,
                                  run_grad_check=False, log_steps=20, distributed=False, save_all_snapshots=True,
        ):      
                self.cfg = cfg
                self.args = args

                # basic settings
                self.epoch = 0
                self.iteration = 0
                self.max_epoch = cfg.optim.max_epoch
               
                self.log_steps = log_steps
                self.distributed = distributed
                self.run_grad_check = run_grad_check
                self.save_all_snapshots = save_all_snapshots

                self.pretrain = args.pretrain
                self.mode = args.mode

                # log
                if self.args.mode == 'train':
                        if args.debug:
                                log_file = osp.join(cfg.log_dir, 'debug-train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
                        else:
                                log_file = osp.join(cfg.log_dir, 'train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
                elif self.args.mode == 'test':
                        if args.debug:
                                log_file = osp.join(cfg.log_dir, 'val-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
                        else:
                                log_file = osp.join(cfg.log_dir, 'debug-val-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
                
                if args.debug:
                        self.pretrain = False
                        self.args.resume = False
                self.logger = Logger(log_file=log_file, local_rank=self.args.local_rank)

                # cuda and distributed
                if not torch.cuda.is_available():
                        raise RuntimeError('No CUDA devices available.')
                self.distributed = self.args.local_rank != -1
                if self.distributed:
                        torch.cuda.set_device(self.args.local_rank)
                        dist.init_process_group(backend='nccl')
                        self.world_size = dist.get_world_size()
                        self.local_rank = self.args.local_rank
                        self.logger.info(f'Using DistributedDataParallel mode (world_size: {self.world_size})')
                else:
                        if torch.cuda.device_count() > 1:
                                self.logger.warning('DataParallel is deprecated. Use DistributedDataParallel instead.')
                        self.world_size = 1
                        self.local_rank = 0
                        self.logger.info('Using Single-GPU mode.')
                self.cudnn_deterministic = cudnn_deterministic
                self.autograd_anomaly_detection = autograd_anomaly_detection
                self.seed = self.cfg.seed + self.local_rank

                initialize(
                        seed=self.seed,
                        cudnn_deterministic=self.cudnn_deterministic,
                        autograd_anomaly_detection=self.autograd_anomaly_detection,
                )
                torch.multiprocessing.set_sharing_strategy('file_system')
                
                # dataset
                if self.mode == 'train':
                        if self.cfg.data.dataset == 'nio':
                                self.logger.info('Using NIO dataset. Now Loading Dataset Pickles')
                                self.train_dataset =  NioVisualLoopDataset(cfg.data.dataset_path, cfg.data.with_amb, \
                                                                                                                        loop_closure_path=cfg.data.loop_pkl_path_train, \
                                                                                                                        img_width=cfg.data.img_width, img_height=cfg.data.img_height,  \
                                                                                                                        dataset_length=cfg.data.dataset_length, neg_num=cfg.data.neg_num, \
                                                                                                                        mode='train')
                        elif self.cfg.data.dataset == 'nclt':
                                self.logger.info('Using NCLT dataset.')
                                from prnet.utils.params import TrainingParams
                                params = TrainingParams(cfg.data.param_path[0], cfg.data.param_path[1])
                                self.train_dataset = NCLTVisualLoopDataset(dataset_path=cfg.data.dataset_path,  \
                                                                                                                           dataset_type='nclt', query_filename=cfg.data.loop_pkl_path_train, params=params)
                        elif self.cfg.data.dataset == 'oxford':
                                self.logger.info('Using Oxford dataset.')
                                from prnet.utils.params import TrainingParams
                                params = TrainingParams(cfg.data.param_path[0], cfg.data.param_path[1])
                                self.train_dataset = OxfordVisualLoopDataset(dataset_path=cfg.data.dataset_path,  \
                                                                                                                           dataset_type='oxford', query_filename=cfg.data.loop_pkl_path_train, params=params)

                        if self.distributed:
                                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.train.batch_size, \
                                                                                                                                collate_fn=SimpleSingleCollateFnPackMode(), pin_memory=False,  \
                                                                                                                                drop_last=False, num_workers=cfg.train.num_workers, sampler=DistributedSampler(self.train_dataset))
                        else:
                                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.train.batch_size, shuffle=True,\
                                                                                                                                collate_fn=SimpleSingleCollateFnPackMode(), pin_memory=False,  \
                                                                                                                                drop_last=False, num_workers=cfg.train.num_workers)
                if self.cfg.data.dataset == 'nio':
                        self.val_dataset =  NioVisualLoopDataset(cfg.data.dataset_path,  loop_closure_path=cfg.data.loop_pkl_path_eval, \
                                                                                                                img_width=cfg.data.img_width, img_height=cfg.data.img_height, mode='eval')
                                                                                              
                # model
                self.register_model(create_model(cfg).cuda())
                
                # optimizer
                optimizer = optim.Adam(self.model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
                self.register_optimizer(optimizer)

                # schedular
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
                self.timer = Timer()
                self.grad_acc_steps = cfg.optim.grad_acc_steps
                
                # tensorboad
                self.summary_board = SummaryBoard(last_n=self.log_steps, adaptive=True)
                self.writer = SummaryWriter(log_dir=cfg.event_dir)

                # loss function and evluation
                if self.mode == 'train':
                        self.loss_func = LossFunction_Total(cfg)

                if self.cfg.data.dataset == 'nio':
                        self.evaluator = LCD_evaluator(self.cfg, self.val_dataset, self.model)
                elif self.cfg.data.dataset == 'nclt':
                        from tools.evaluate import GLEvaluator
                        self.evaluator = GLEvaluator(cfg.data.dataset_path, 'nclt', cfg.data.loop_pkl_path_eval, device='cuda', params=self.cfg, radius=[2, 5, 10, 20], k=20, n_samples=None)
                elif self.cfg.data.dataset == 'oxford':
                        from tools.evaluate import GLEvaluator
                        self.evaluator = GLEvaluator(cfg.data.dataset_path, 'oxford', cfg.data.loop_pkl_path_eval, device='cuda', params=self.cfg, radius=[2, 5, 10, 20], k=20, n_samples=None)                        
                
                self.snapshot_dir = cfg.snapshot_dir
                if self.args.finetune:
                        self.args.pretrain = True
                        self.args.resume = True
                if self.args.resume:
                        self.load_snapshot(osp.join(self.cfg.snapshot_dir, 'snapshot.pth.tar'), pretrain=self.args.pretrain) # for nio resume
                        # self.load_snapshot(osp.join(self.cfg.snapshot_dir, 'v_0_0_4.pth.tar'), pretrain=args.pretrain) # for nclt gem
                        # self.load_snapshot(osp.join(self.cfg.snapshot_dir, 'v_disco.pth.tar'), pretrain=args.pretrain) # for nclt disco

        def register_model(self, model):
                if self.distributed:
                        local_rank = self.local_rank
                        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
                self.model = model
                message = 'Model description:\n' + str(model)
                self.logger.info(message)
                return model

        def register_optimizer(self, optimizer):
                if self.distributed:
                        for param_group in optimizer.param_groups:
                                param_group['lr'] = param_group['lr'] * self.world_size
                self.optimizer = optimizer

        def set_train_mode(self):
                self.training = True
                self.model.train()
                self.model.training = True
                torch.set_grad_enabled(True)

        def set_eval_mode(self):
                self.training = False
                self.model.eval()
                self.model.traning = False
                torch.set_grad_enabled(False)
        
        def optimizer_step(self, iteration):
                if iteration % self.grad_acc_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
        
        def get_lr(self):
                return self.optimizer.param_groups[0]['lr']

        def write_event(self, phase, event_dict, index):
                if self.local_rank != 0:
                        return
                for key, value in event_dict.items():
                        self.writer.add_scalar(f'{phase}/{key}', value, index)
        
        def release_tensors(self, result_dict):
                if self.distributed:
                        result_dict = all_reduce_tensors(result_dict, world_size=self.world_size)
                        result_dict = release_cuda(result_dict)
                return result_dict

        def load_snapshot(self, snapshot, fix_prefix=True, pretrain=False):
                self.logger.info('Loading from "{}".'.format(snapshot))
                state_dict = torch.load(snapshot, map_location=torch.device('cpu'))

                # Load model
                model_dict = state_dict['model']
                if fix_prefix and self.distributed:
                        model_dict = OrderedDict([('module.' + key, value) for key, value in model_dict.items()])
                self.model.load_state_dict(model_dict, strict=False)

                # log missing keys and unexpected keys
                snapshot_keys = set(model_dict.keys())
                model_keys = set(self.model.state_dict().keys())
                missing_keys = model_keys - snapshot_keys
                unexpected_keys = snapshot_keys - model_keys
                if self.distributed:
                        missing_keys = set([missing_key[7:] for missing_key in missing_keys])
                        unexpected_keys = set([unexpected_key[7:] for unexpected_key in unexpected_keys])
                if len(missing_keys) > 0:
                        message = f'Missing keys: {missing_keys}'
                        self.logger.warning(message)
                if len(unexpected_keys) > 0:
                        message = f'Unexpected keys: {unexpected_keys}'
                        self.logger.warning(message)
                self.logger.info('Model has been loaded.')

                # Load other attributes
                if not pretrain:
                        if 'epoch' in state_dict:
                                self.epoch = state_dict['epoch']
                                self.logger.info('Epoch has been loaded: {}.'.format(self.epoch))
                        if 'iteration' in state_dict:
                                self.iteration = state_dict['iteration']
                                self.logger.info('Iteration has been loaded: {}.'.format(self.iteration))
                        if 'optimizer' in state_dict and self.optimizer is not None:
                                self.optimizer.load_state_dict(state_dict['optimizer'])
                                self.logger.info('Optimizer has been loaded.')
                        if 'scheduler' in state_dict and self.scheduler is not None:
                                self.scheduler.load_state_dict(state_dict['scheduler'])
                                self.logger.info('Scheduler has been loaded.')

        def save_snapshot(self, filename):
                if self.local_rank != 0:
                        return

                model_state_dict = self.model.state_dict()
                # Remove '.module' prefix in DistributedDataParallel mode.
                if self.distributed:
                        model_state_dict = OrderedDict([(key[7:], value) for key, value in model_state_dict.items()])

                # save model
                filename = osp.join(self.snapshot_dir, filename)
                state_dict = {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'model': model_state_dict,
                }
                torch.save(state_dict, filename)
                self.logger.info('Model saved to "{}"'.format(filename))

                # save snapshot
                snapshot_filename = osp.join(self.snapshot_dir, 'snapshot.pth.tar')
                state_dict['optimizer'] = self.optimizer.state_dict()
                if self.scheduler is not None:
                        state_dict['scheduler'] = self.scheduler.state_dict()
                torch.save(state_dict, snapshot_filename)
                self.logger.info('Snapshot saved to "{}"'.format(snapshot_filename))

        def train_epoch(self):
                if self.distributed:
                        self.train_loader.sampler.set_epoch(self.epoch)
                total_iterations = len(self.train_loader)
                self.optimizer.zero_grad()
                for iteration, data_dict in enumerate(self.train_loader):
                        self.inner_iteration = iteration + 1
                        self.iteration += 1
                        data_dict = to_cuda(data_dict)

                        self.timer.add_prepare_time()
                        try:
                                output_dict = self.model(data_dict)
                        except:
                                print('CUDA out of memory!')
                                continue
                        loss_dict = self.loss_func(output_dict, data_dict)
                        loss_dict['loss'].backward()
                        result_dict, hard_idxs = compute_metrics(output_dict, data_dict, margin=self.cfg.loss.margin_triplet)
                        result_dict.update(loss_dict)
                        self.train_dataset.update_neg_list(hard_idxs)
                        self.optimizer_step(iteration + 1)
                        self.timer.add_process_time()

                        result_dict = release_cuda(result_dict)
                        self.summary_board.update_from_result_dict(result_dict)

                        if self.inner_iteration % self.log_steps == 0:
                                summary_dict = self.summary_board.summary()
                                message = get_log_string(
                                        result_dict=summary_dict,
                                        epoch=self.epoch,
                                        max_epoch=self.max_epoch,
                                        iteration=self.inner_iteration,
                                        max_iteration=total_iterations,
                                        lr=self.get_lr(),
                                        timer=self.timer
                                )
                                self.logger.info(message)
                                self.write_event('train', summary_dict, self.iteration)

                        torch.cuda.empty_cache()

                message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)
                self.logger.critical(message)
                if self.scheduler is not None:
                        self.scheduler.step()
                self.save_snapshot(f'epoch-{self.epoch}.pth.tar')
                if not self.save_all_snapshots:
                        last_snapshot = f'epoch-{self.epoch - 1}.pth.tar'
                        if osp.exists(last_snapshot):
                                os.remove(last_snapshot)

        def eval_epoch(self):
                self.set_eval_mode()
                self.evaluator.model = self.model
                summary_board = SummaryBoard(adaptive=True)
                timer = Timer()

                if self.cfg.data.dataset == 'nio':
                        total_iterations = len(self.val_dataset)
                        pbar = tqdm.tqdm(enumerate(self.val_dataset), total=total_iterations)
                        place_names = list(self.val_dataset.loop_closures.keys())
                        for iteration, data_dict in pbar:
                                self.inner_iteration = iteration + 1
                                # if self.inner_iteration > 1 and self.mode == 'train':
                                #         break
                                timer.add_prepare_time()

                                result_dict = self.evaluator.run(data_dict, place_names[iteration])

                                torch.cuda.synchronize()
                                timer.add_process_time()
                                result_dict = self.release_tensors(result_dict)
                                summary_board.update_from_result_dict(result_dict)

                                inner_iteration = iteration + 1
                                message = get_log_string(
                                        result_dict=summary_board.summary(),
                                        epoch=self.epoch,
                                        iteration=self.inner_iteration,
                                        max_iteration=total_iterations,
                                        timer=timer,
                                )
                                pbar.set_description(message)
                                torch.cuda.empty_cache()
                elif self.cfg.data.dataset == 'nclt' or self.cfg.data.dataset == 'oxford':
                        timer.add_prepare_time()
                        result_dict = self.evaluator.run()
                        torch.cuda.synchronize()
                        timer.add_process_time()
                        result_dict = self.release_tensors(result_dict)
                        summary_board.update_from_result_dict(result_dict)
                        message = get_log_string(
                                result_dict=summary_board.summary(),
                                epoch=self.epoch,
                                timer=timer,
                        )
                        torch.cuda.empty_cache()

                summary_dict = summary_board.summary()
                message = '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
                self.logger.critical(message)
                self.write_event('val', summary_dict, self.epoch)
                self.set_train_mode()

        def run(self):
                self.set_train_mode()
                if self.args.mode == 'test':
                        self.eval_epoch()
                elif self.args.mode == 'train':
                        while self.epoch < self.max_epoch:
                                self.epoch += 1
                                self.train_epoch()
                                self.eval_epoch()

if __name__ == '__main__':
        # set up trainner
        args = parser.parse_args()
        if args.debug:
                from configs.cfg_debug import make_cfg
        elif args.dataset == 'nio':
                if args.finetune:
                        from configs.cfg_joint_finetune import make_cfg
                else:
                        from configs.cfg_joint_baseline import make_cfg
        elif args.dataset == 'nclt':
                from configs.cfg_joint_baseline_nclt import make_cfg
        elif args.dataset == 'oxford':
                from configs.cfg_joint_baseline_oxford import make_cfg
        cfg = make_cfg()
        trainer = Trainer(cfg, args)
        trainer.run()