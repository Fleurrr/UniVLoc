# UniVLoc
Codes for paper <UniVLoc: Towards Unified Visual Localization> submitted to **IEEE International Conference on Multimedia&Expo 2025 (ICME 2025)**
  

## 1.Environments
Tested on cuda11.4, python 3.9
```
# install torch-related dependencies first
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
# compiling dependences
cd ops
python setup.py develop
cd ../
# install other requirement
pip install -r requirements.txt
```

  

## 2.Data Preparation
### 2.1 NCLT Dataset
Download [NCLT](https://robots.engin.umich.edu/nclt/) Dataset 
Experiments in the paper were done on 2012-02-04 and 2012-03-17
```
cd /prnet/datasets/nclt
python image_preprocess.py --dataset_root=/path/to/nclt/data/
python generate_training_tuples.py --dataset_root=/path/to/nclt/data/
python generate_evaluation_sets.py --dataset_root=/path/to/nclt/data/
```
Folder structure of the processed data
```
--NCLT
  |--2012-02-04
     |--lb3
     |--lb3_u
     |--lb3_u_384
        |--Cam1
        |--Cam2
        |--Cam3
        |--Cam4
        |--Cam5
     |--velodyne_sync
  |--2012-03-17
  |--cam_params
     |--image_meta.pkl
  |--test_2012-02-04_2012-03-17_0.2.pickle
  |--train_2012-02-04_2012-03-17_2.0_3.0.pickle 
  |--val_2012-02-04_2012-03-17_2.0_3.0.pickle    
```

### 2.2 Oxford Dataset
Download [Oxford](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets) Dataset
Experiments in the paper were done on 2019-01-15-13-06-37 and 2019-01-11-13-24-51
```
cd /prnet/datasets/oxford
python image_preprocess.py --dataset_root=/path/to/oxford/data/
python generate_training_tuples.py --dataset_root=/path/to/oxford/data/
python generate_evaluation_sets.py --dataset_root=/path/to/oxford/data/
```
Folder structure of the processed data
```
|--oxford
   |--2019-01-11-13-24-51-radar-oxford-10k
   |--2019-01-15-13-06-37-radar-oxford-10k
   |--image_meta.pkl
   |--test_2019-01-11-13-24-51-radar-oxford-10k_2019-01-15-13-06-37-radar-oxford-10k_0.2.pickle
   |--train_2019-01-11-13-24-51-radar-oxford-10k_2019-01-15-13-06-37-radar-oxford-10k_2.0_3.0.pickle
   |--val_2019-01-11-13-24-51-radar-oxford-10k_2019-01-15-13-06-37-radar-oxford-10k_2.0_3.0.pickle
```

## 2.Training and Evaluation

### 2.1 Training
```
# Single Card
python main.py --local_rank=-1 --dataset=nclt #oxford
# multi-card
# set local_rank=nGPU first, 4 in example
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```
Parameter Description for ``main.py``
```
--resume: Continue training from the last interrupted training task. The path of the last training task depends on config, and the default is ../output/xxx/snapshots/snapthot.pth.tar
--pretrain: Use the result of the last training task as the pre-trained model and start training from scratch. Use it with --resume
--finetune: Use the fine-tuning mode to fine-tune the result of the last training task. By default, both --resume and --pretrain are enabled
--debug: debug mode, which can be used for training and testing in the workshop
--mode: The default is train, which means training the model. It can be set to test for testing on the test set. Generally, --resume and --pretrain need to be enabled during testing
--dataset: Training on two public datasets is supported, namely nclt and oxford
```
### 2.1 Testing
```
python main.py --local_rank=-1 --mode=test --dataset=nclt #oxford --resume --pretrain 
```
