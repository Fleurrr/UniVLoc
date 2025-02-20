from configs.cfg_place_recognition_baseline import make_cfg
from dataset.collate import SimpleSingleCollateFnPackMode
from dataset.nio_visual_loop_dataset import NioVisualLoopDataset
from torch.utils.data import Dataset, DataLoader
from model.place_recognition_branch import  MCVPR
from utils.torch_utils import to_cuda, all_reduce_tensors, release_cuda, initialize
from eval.evaluator import LCD_evaluator
import pdb

import numpy as np
import math
import cv2

from utils.geometry import apply_transform, inverse_transform

from prnet.utils.params import TrainingParams
from dataset.nclt_visual_loop_dataset import NCLTVisualLoopDataset
from dataset.oxford_visual_loop_dataset import OxfordVisualLoopDataset

EPS = 1e-6

def save_imgs(img, prefix):
    cv2.imwrite('./check/' + prefix + '_fw.png', np.transpose(img[0], [1,2,0]))
    cv2.imwrite('./check/' + prefix + '_rn.png', np.transpose(img[1], [1,2,0]))
    cv2.imwrite('./check/' + prefix + '_left.png', np.transpose(img[2], [1,2,0]))
    cv2.imwrite('./check/' + prefix + '_right.png', np.transpose(img[3] , [1,2,0]))
    return

def draw_pc_in_image(points_cam, img, color=(0, 0, 255)):
    H, W = img.shape[0], img.shape[1]
    for i in range(points_cam.shape[0]):
        py, px, pz = int(points_cam[i][1] / (points_cam[i][2] + EPS)), int(points_cam[i][0] / (points_cam[i][2] + EPS)), int(points_cam[i][2])
        if  py < 0 or py >= H or px < 0 or px >= W or pz <= 0:
            continue
        else:
            # cv2.circle(img, (px, py), 1, color, 1) 
            img[py, px] = (int(255 * py / H), 0, int(255 * py / H))
    return img

def transform_points_to_img(P, K, imgs, prefix='qry'):
    # # points_vehicle = np.array([[-1, -0.5, 1.5], [-1, 0, 1.5], [-1, 0.5, 1.5], [1, -0.5, 1.5], [1, 0, 1.5], [1, 0.5, 1.5],  [0,-0.5, 1.5], [0, 0, 1.5], [0, 0.5, 1.5]]) * 2 #(N, 3)-xyz
    x_values = np.arange(-15, 15, 30/100)  # Sample from -5 to 5 with step size 0.5
    y_values = np.arange(-15, 15, 30/100)  # Sample from -5 to 5 with step size 0.5
    z_values = np.arange(-2, 3, 0.5)  # Sample from 0 to 10 with step size 0.5
    x, y, z = np.meshgrid(x_values, y_values, z_values, indexing='ij')
    points_vehicle = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    for cam_id in range(P.shape[0]):
        points_cam = apply_transform(points_vehicle, P[cam_id, ...])
        points_cam = np.transpose(np.matmul(K[cam_id, ...], np.transpose(points_cam, [1,0])), [1,0])
    
        img = draw_pc_in_image(points_cam, np.transpose(imgs[cam_id, ...], [1,2,0]).copy())
        cv2.imwrite(prefix + '_' + str(cam_id) + '_points_proj.png', img)
        # cv2.imwrite(prefix + '_' + str(cam_id) + '_points_proj.png', np.transpose(imgs[cam_id, ...], [1,2,0]).copy())
    return

def IPM(P, K, imgs, prefix='qry'):
    ipm = np.zeros((300, 300, 3))
    for idx in range(300):
        for idy in range(300):
            points_vehicle = np.array([-15 + idx * 0.1, -15 + idy * 0.1, 0])[np.newaxis,...]
            for cam_id in range(P.shape[0]):
                points_cam = apply_transform(points_vehicle, P[cam_id, ...])
                points_cam = np.transpose(np.matmul(K[cam_id, ...], np.transpose(points_cam, [1,0])), [1,0])
                py, px, pz = points_cam[0][1] / (points_cam[0][2] + EPS), points_cam[0][0] / (points_cam[0][2] + EPS), points_cam[0][2]

                img = np.transpose(imgs[cam_id, ...], [1,2,0])
                H, W = img.shape[0], img.shape[1]
                if py < 0 or py >= H or px < 0 or px >= W or pz <= 0:
                    continue                
                else:
                    y0 = int(math.floor(py))  
                    y1 = min(y0 + 1, H-1)  
                    x0 = int(math.floor(px))  
                    x1 = min(x0 + 1, W-1)
                    Q11 = img[y0, x0]  
                    Q21 = img[y1, x0]  
                    Q12 = img[y0, x1]  
                    Q22 = img[y1, x1]
                    wa = (px - x0) * ((y1 - py) / (y1 - y0) if y0 != y1 else 1.0)  
                    wb = (px - x0) * ((py - y0) / (y1 - y0) if y0 != y1 else 1.0)  
                    wc = (x1 - px) * ((y1 - py) / (y1 - y0) if y0 != y1 else 1.0)  
                    wd = (x1 - px) * ((py - y0) / (y1 - y0) if y0 != y1 else 1.0)
                    pixel = (wa * Q11 + wb * Q21 + wc * Q12 + wd * Q22).astype(int)                     

                    ipm[idx, idy] = pixel
    cv2.imwrite(prefix + '.png', ipm)
    return

def train_model_with_real_data():
    dataset = NioVisualLoopDataset(dataset_path='../data/training/',  data_info_path='./data/data_info.txt', loop_closure_path='./dataset/loop_closure_5_30_140_amb_limited_sample.pkl', dataset_length=3000, mode='train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=SimpleSingleCollateFnPackMode(), num_workers=1)
    pdb.set_trace()
    cfg = make_cfg()
    model = MCVPR(cfg).cuda()
    for iteration, data_dict in enumerate(dataloader):
        # 'positive_imgs', 'negative_P', 'positive_P', 'query_K', 'negative_K', 'query_P', 'relative_rotation', 'query_imgs', 'negative_imgs', 'positive_K', 'relative_translation', 'batch_size'
        # 'positive_imgs', 'negative_imgs', 'query_imgs' : (B, S, H, W, 3)
        # 'positive_P', 'negative_P', 'query_P': (B, S, 4, 4)
        # 'positive_K', 'negative_K', 'query_K': (B, S, 3, 3)
        # 'relative_rotation': (B, 3, 3)
        # 'relative_translation': (B, 3, )
        qry_data_dict = {'query_img':  data_dict['query_imgs'], 'query_P':  data_dict['query_P'], 'query_K':  data_dict['query_K']}
        # pos_data_dict = {'query_img':  data_dict['positive_imgs'], 'query_P':  data_dict['positive_P'], 'query_K':  data_dict['positive_K']}
        # neg_data_dict = {'query_img':  data_dict['negative_imgs'], 'query_P':  data_dict['negative_P'], 'query_K':  data_dict['negative_K']}

        qry_desc= model(to_cuda(qry_data_dict)) #(b, 1024)
        break
    return

def eval_model_with_real_data():
    dataset = NioVisualLoopDataset(dataset_path='./data/evaling/', loop_closure_path='./dataset/loop_closure_eval.pkl', mode='eval')
    cfg = make_cfg()
    model = MCVPR(cfg).cuda()
    evaluator = LCD_evaluator(dataset, model)
    for place_id, place_dict in enumerate(dataset):
        if bool(place_dict):
            res = evaluator.run(place_dict)
            pdb.set_trace()
    return

def check_intrinscics_and_extrinsics():
    dataset = NioVisualLoopDataset(dataset_path='/map-hl/gary.huang/visual_loop_closure_dataset/finetune', loop_closure_path='./dataset/loop_closure_o2_i2_n3_U_l700_finetune.pkl') #, img_width=427 // 2, img_height=240 // 2, )
    # # # dict_keys(['query_imgs', 'positive_imgs', 'negative_imgs', 'query_K', 'positive_K', 'negative_K', 'query_P', 'positive_P', 'negative_P', 'relative_rotation', 'relative_translation'])
    # pdb.set_trace()
    # params = TrainingParams('./prnet/config/deformable/config_deformable.txt', './prnet/config/deformable/deformable.txt')
    # dataset = NCLTVisualLoopDataset(dataset_path='/map-hl/gary.huang/public_dataset/NCLT/',  dataset_type='nclt', query_filename='train_2012-02-04_2012-03-17_2.0_10.0.pickle', params=params)
    # params = TrainingParams('./prnet/config/deformable/config_deformable.txt', './prnet/config/deformable/deformable.txt')
    # dataset = OxfordVisualLoopDataset(dataset_path='/map-hl/gary.huang/public_dataset/Oxford/dataset/',  \
    #                                                         dataset_type='oxford', query_filename='train_2019-01-11-13-24-51-radar-oxford-10k_2019-01-15-13-06-37-radar-oxford-10k_2.0_3.0.pickle', params=params)
    for i in range(len(dataset)):
        data_dict = dataset[i]
        imgs =   data_dict['query_imgs'] #(s, c, h, w) - [fw_img, rn_img, svc_left_img, svc_right_img]
        P =  data_dict['query_P'] # (s, 4, 4) 
        K =  data_dict['query_K'] # (s ,3, 3) 

        imgs = imgs - imgs.min()
        imgs = imgs * 255 / imgs.max()
        imgs = imgs[:, [0,1,2], :, :]

        pos_imgs =  data_dict['positive_imgs'] #(s, c, h, w) - [fw_img, rn_img, svc_left_img, svc_right_img]
        pos_P =  data_dict['positive_P'] # (s, 4, 4) 
        pos_K =  data_dict['positive_K'] # (s ,3, 3) 

        neg_imgs = data_dict['negative_imgs']
        neg_P = data_dict['negative_P']
        neg_K = data_dict['negative_K']

        pos_imgs = pos_imgs - pos_imgs.min()
        pos_imgs = pos_imgs * 255 / pos_imgs.max()
        pos_imgs = pos_imgs[:, [0,1,2], :, :]

        neg_imgs = neg_imgs - neg_imgs.min()
        neg_imgs = neg_imgs * 255 / neg_imgs.max()
        neg_imgs = neg_imgs[0]
        neg_imgs = neg_imgs[:, [0, 1, 2], :, :]

        print(data_dict['relative_euler'])
        print(data_dict['relative_translation'])
        relative_euler = data_dict['relative_euler']

        save_imgs(imgs, 'qry')
        save_imgs(pos_imgs, 'pos')
        save_imgs(neg_imgs, 'neg')
        import pdb; pdb.set_trace()
        # if abs(data_dict['relative_euler'][2]) > 20 and abs(data_dict['relative_euler'][2]) < 90: # and abs(data_dict['relative_translation'][0]) < 3 and abs(data_dict['relative_translation'][1]) < 3:
        #     transform_points_to_img(P, K, imgs, prefix='qry')
        #     transform_points_to_img(pos_P, pos_K, pos_imgs, prefix='pos')
        #     pdb.set_trace()
            

        # transform_points_to_img(P, K, imgs, prefix='qry')
        # transform_points_to_img(pos_P, pos_K, pos_imgs, prefix='pos')
        # pdb.set_trace()


    # IPM(P, K, imgs, prefix='qry')
    # IPM(pos_P, pos_K, pos_imgs, prefix='pos')
    return

def check_bev_mask():
    bev_mask = np.load('./bev_mask.npy').reshape(4, 100, 100, 20)
    cam_ids = [0, 1, 2, 3]
    for cam_id in cam_ids:
        for i in range(bev_mask.shape[3]):
            img = bev_mask[cam_id, :,:, i].astype(int) * 255
            img_name = './output/' + str(cam_id) + '_' + str(i) + '.png'
            cv2.imwrite(img_name, img)
    return

if __name__ == '__main__':
    # eval_model_with_real_data()
    # train_model_with_real_data()
    check_intrinscics_and_extrinsics()
    # check_bev_mask()
    
