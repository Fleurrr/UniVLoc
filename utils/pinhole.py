import cv2
import numpy as np
import math
import json
from scipy.spatial.transform import Rotation

def scale_intrinsics(K, sx, sy):
    pose_aug = np.eye(3)
    pose_aug[0, 0] = sx
    pose_aug[1, 1] = sy
    K = pose_aug @ K
    return K

def read_json_from_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def fast_pinhole_distortion(distort_img, param):
    result_image_width = 854
    result_image_height = 480
    distortion_coeffcients = np.array(param['distortion_coeffcients'])
    camera_matrix = np.array(param['camera_matrix'])

    distort_img = cv2.resize(distort_img, (param['camera_width'], param['camera_height']))
    dw, dh = param['camera_width']/result_image_width, param['camera_height']/result_image_height
    new_camera_matrix = scale_intrinsics(camera_matrix, 1/dw, 1/dh)

    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffcients, None, new_camera_matrix, (result_image_width, result_image_height), 5)
    undistort_img = cv2.remap(distort_img, mapx, mapy, cv2.INTER_LINEAR)

    return undistort_img, new_camera_matrix

if __name__ == '__main__':
    img = cv2.imread('./data/undistort/sessions/kx-100012466-94d73d5eff4046c50508729570001010-1703934003/1703934003_kx-100012466_UnknownVehicle_unknown_0_common/images/FW/key_frame_20.jpg')
    cv2.imwrite('./distort.jpg', img) 
    param = read_json_from_file('./data/undistort/sessions/kx-100012466-94d73d5eff4046c50508729570001010-1703934003/1703934003_kx-100012466_UnknownVehicle_unknown_0_common/parameters/camera/front_wide/front_wide.json')
    undistort_img, __getitem__ = fast_pinhole_distortion(img, param['intrinsic_param'])
    cv2.imwrite('./undistort.jpg', undistort_img) 