import cv2
import numpy as np
import math
import json
from scipy.spatial.transform import Rotation

def load_fisheye_json(json_path, cam_name='svc_left'):
    out_dict = {}
    try:
        with open(json_path, 'r') as file:
            cam = json.load(file)

        # cam_name
        cam_name_ = cam["name"]
        out_dict["cam_name"] = cam_name

        # intrinsic_param
        intrinsic_param = cam["intrinsic_param"]

        if "ImgSize" in cam and cam["ImgSize"] is not None:
            width_ = int(cam["ImgSize"][0])
            height_ = int(cam["ImgSize"][1])
        else:
            width_ = int(intrinsic_param["camera_width"])
            height_ = int(intrinsic_param["camera_height"])
        
        out_dict["width_"] = width_
        out_dict["height_"] = height_

        len_pol_pixel2cam_ = int(intrinsic_param["cam2world_len"])
        len_pol_cam2pixel_ = int(intrinsic_param["world2cam_len"])
        
        out_dict['len_pol_cam2pixel_'] = len_pol_cam2pixel_
        out_dict['len_pol_pixel2cam_'] = len_pol_pixel2cam_

        pol_pixel2cam_ = np.array(intrinsic_param["cam2world"][:len_pol_pixel2cam_])
        pol_cam2pixel_ = np.array(intrinsic_param["world2cam"][:len_pol_cam2pixel_])

        out_dict["pol_pixel2cam_"] = pol_pixel2cam_
        out_dict["pol_cam2pixel_"] = pol_cam2pixel_
        
        affine_matrix_ = np.array([
            [1.0, float(intrinsic_param["affine_e"])],
            [float(intrinsic_param["affine_d"]), float(intrinsic_param["affine_c"])]
        ])

        principal_pt_ = np.array(intrinsic_param["center"])
        
        out_dict["affine_matrix_"] = affine_matrix_
        out_dict["principal_pt_"] = principal_pt_

        # extrinsic_param
        extrinsic_param = cam["extrinsic_param"]
        t_cam2veh_ = np.array(extrinsic_param["translation"])
        rvec = np.array(extrinsic_param["rotation"])
        r = Rotation.from_rotvec(rvec)
        rvec_cam2veh_ = r.as_rotvec()
        
        out_dict["rvec_cam2veh_"] = rvec_cam2veh_

        q_cam2veh_ = r.as_quat()
        q_veh2cam_ = q_cam2veh_.conj()
        
        out_dict["q_veh2cam_"] = q_veh2cam_

        trans_cam2veh_ = np.eye(4)
        trans_cam2veh_[:3, :3] = r.as_matrix()
        trans_cam2veh_[:3, 3] = t_cam2veh_

        trans_veh2cam_ = np.linalg.inv(trans_cam2veh_)
        
        out_dict["trans_veh2cam_"] = trans_veh2cam_

        # print(f"read svc calib: {cam_name}")
        # print(trans_cam2veh_)
        return out_dict

    except FileNotFoundError:
        print(f"Cannot open bev json file, path: {json_path}")
        return False

def cam2pixel(cam_pt, param):
    len_pol_cam2pixel_ = param['len_pol_cam2pixel_']
    pol_cam2pixel_ = param['pol_cam2pixel_']
    affine_matrix_= param['affine_matrix_']
    principal_pt_ = param['principal_pt_']
    
    radius = np.sqrt(cam_pt[0] ** 2 + cam_pt[1] ** 2)
    if radius == 0:
        radius = 1e-10
    theta = np.arctan2(-cam_pt[2], radius)
    r_pow = np.array([theta ** i for i in range(len_pol_cam2pixel_)])
    g_theta = np.dot(pol_cam2pixel_, r_pow)
    uv = np.array([g_theta * cam_pt[0] / radius, g_theta * cam_pt[1] / radius])
    pixel_pt = np.dot(affine_matrix_, uv) + principal_pt_
    return pixel_pt

def fast_cam2pixel(cam_pt, param):
    len_pol_cam2pixel_ = param['len_pol_cam2pixel_']
    pol_cam2pixel_ = param['pol_cam2pixel_']
    affine_matrix_= param['affine_matrix_']
    principal_pt_ = param['principal_pt_']
    
    radius = np.sqrt(cam_pt[:, :, 0] ** 2 + cam_pt[:, :, 1] ** 2) #(w, h)
    radius[radius == 0] = 1e-10 #(w, h)
    theta = np.arctan2(-cam_pt[:,:,2], radius) #(w, h)
    r_pow = np.transpose(np.array([theta ** i for i in range(len_pol_cam2pixel_)]), (1,2,0) ) #(w, h, 12)
    g_theta = np.dot(r_pow, pol_cam2pixel_) #(w, h)
    u = g_theta * cam_pt[:,:,0] / radius
    v = g_theta * cam_pt[:,:,1] / radius
    uv = np.concatenate([u[:,:,np.newaxis], v[:,:,np.newaxis]], -1) #(w, h, 2)
    pixel_pt = np.dot(uv, affine_matrix_) + principal_pt_ #(w, h, 2)
    return pixel_pt

def fisheye_distortion(distort_img, param, result_image_width=854, result_image_height=480):
    plane_width = 16.0
    plane_height = 12.0
    plane_z = 3.0
    deleta_w = plane_width / (result_image_width - 1)
    deleta_h = plane_height / (result_image_height - 1)
    distort_img = cv2.resize(distort_img, (param["width_"], param["height_"]))
    
    undistort_img = np.zeros((result_image_height, result_image_width, 3), dtype=np.uint8)
    
    for ih in range(result_image_height):
        for iw in range(result_image_width):
            plane_pt = np.array([iw * deleta_w, ih * deleta_h, 0])
            camera_pt = plane_pt + np.array([-plane_width / 2, -plane_height / 2.0, plane_z])
            pixel_pt = cam2pixel(camera_pt, param)
            u1, v1 = int(pixel_pt[0]), int(pixel_pt[1])

            if u1 < 0 or u1 >= distort_img.shape[1] - 1 or v1 < 0 or v1 >= distort_img.shape[0] - 1:
                continue

            ind = ih * result_image_width + iw

            for k in range(3):
                undistort_img[ind // result_image_width, ind % result_image_width, k] = distort_img[v1, u1, k]
    return undistort_img

def fast_fisheye_distortion(distort_img, param, result_image_width=854, result_image_height=480):
    plane_width = 16.0
    plane_height = 12.0
    plane_z = 3.0

    scale_x = result_image_width / plane_width
    scale_y = result_image_height / plane_height

    new_camera_matrix = np.array([[scale_x * plane_z, 0., result_image_width / 2], [0., scale_y * plane_z, result_image_height / 2], [0., 0., 1.]])

    deleta_w = plane_width / (result_image_width - 1)
    deleta_h = plane_height / (result_image_height - 1)

    distort_img = cv2.resize(distort_img, (param["width_"], param["height_"]))
    undistort_img = np.zeros((result_image_height, result_image_width, 3), dtype=np.uint8)
    
    grid_w, grid_h = np.meshgrid(np.arange(result_image_width) * deleta_w, np.arange(result_image_height) * deleta_h, indexing='ij')
    plane_pt = np.stack((grid_w, grid_h, np.zeros_like(grid_h)), axis=-1) #(w, h, 3)
    camera_pt = plane_pt + np.array([-plane_width / 2.0, -plane_height / 2.0, plane_z]) #(w, h, 3)
    pixel_pt = fast_cam2pixel(camera_pt, param) #(w, h, 2)
    u1, v1 = pixel_pt[:, :, 0].astype(int), pixel_pt[:, :, 1].astype(int) #(w, h), (w, h)
    undistort_img = np.transpose(distort_img[v1, u1], [1,0,2])
    
    return undistort_img, new_camera_matrix


if __name__ == '__main__':
    img = cv2.imread('./data/undistort/sessions/kx-100012466-94d73d5eff4046c50508729570001010-1703934003/1703934003_kx-100012466_UnknownVehicle_unknown_0_common/images/SVC-Left/key_frame_20.jpg')
    cv2.imwrite('./distort.jpg', img) 
    param = load_fisheye_json('./data/undistort/sessions/kx-100012466-94d73d5eff4046c50508729570001010-1703934003/1703934003_kx-100012466_UnknownVehicle_unknown_0_common/parameters/camera/svc_left/svc_left.json')
    undistort_img, K = fast_fisheye_distortion(img, param)
    print(K)
    cv2.imwrite('./undistort.jpg', undistort_img) 