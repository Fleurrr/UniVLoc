from typing import Tuple, List, Optional, Union, Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch


# Basic Utilities


def get_nearest_neighbor(
    q_points: np.ndarray,
    s_points: np.ndarray,
    return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1, n_jobs=-1)
    if return_index:
        return distances, indices
    else:
        return distances


def regularize_normals(points, normals, positive=True):
    r"""Regularize the normals towards the positive/negative direction to the origin point.

    positive: the origin point is on positive direction of the normals.
    negative: the origin point is on negative direction of the normals.
    """
    dot_products = -(points * normals).sum(axis=1, keepdims=True)
    direction = dot_products > 0
    if positive:
        normals = normals * direction - normals * (1 - direction)
    else:
        normals = normals * (1 - direction) - normals * direction
    return normals


# Transformation Utilities


def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points

def apply_transform_torch(points, transform):
    # points: (B, N, 3), transform: (B, 4, 4)
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3]
    points = torch.matmul(points, rotation.transpose(2,1)) + translation.unsqueeze(1).repeat(1, points.shape[1], 1)
    return points

def cam_to_pixel_torch(points, K):
    # points: (B, N, 3), K:(B, 3, 3)
    points = torch.transpose(torch.matmul(K, torch.transpose(points, 2, 1)), 2, 1) #(B, N, 3)
    points[:,:,0] = points[:,:,0] / points[:,:,2]
    points[:,:,1] = points[:,:,1] / points[:,:,2]
    return points

def apply_transform_ndt(map_pos: np.ndarray, map_cov: np.ndarray, transform: np.ndarray):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    map_pos = np.matmul(map_pos, rotation.T) + translation
    cov_1, cov_2, cov_3 = np.expand_dims(map_cov[:,:3], axis=-1), np.expand_dims(map_cov[:,3:6], axis=-1), np.expand_dims(map_cov[:,6:], axis=-1)
    map_cov =  np.concatenate([cov_1, cov_2, cov_3], axis = -1)
    map_cov = np.matmul(np.matmul(rotation, map_cov), rotation.T)
    map_cov = np.concatenate([map_cov[:,:,0], map_cov[:,:,1], map_cov[:,:,2]], axis = 1)
    return map_pos, map_cov

def compose_transforms(transforms: List[np.ndarray]) -> np.ndarray:
    r"""
    Compose transforms from the first one to the last one.
    T = T_{n_1} \circ T_{n_2} \circ ... \circ T_1 \circ T_0
    """
    final_transform = transforms[0]
    for transform in transforms[1:]:
        final_transform = np.matmul(transform, final_transform)
    return final_transform

def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    rm = Rotation.from_quat(quaternion)
    rotation_matrix = rm.as_matrix()
    return rotation_matrix
    
def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    R_quat = Rotation.from_matrix(rotation).as_quat()
    return R_quat

def rotation_to_euler(rotation: np.ndarray) -> np.ndarray:
    R_euler = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
    return R_euler

def euler_to_rotation(euler):
    r = Rotation.from_euler('xyz', [euler[0], euler[1], euler[2]], degrees=True)
    R_mat = r.as_matrix()
    return R_mat

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    r"""Inverse rigid transform.

    Args:
        transform (array): (4, 4)

    Return:
        inv_transform (array): (4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)  # (3, 3), (3,)
    inv_rotation = rotation.T  # (3, 3)
    inv_translation = -np.matmul(inv_rotation, translation)  # (3,)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)  # (4, 4)
    return inv_transform

def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation

def random_sample_rotation_v2() -> np.ndarray:
    axis = np.random.rand(3) - 0.5
    axis = axis / np.linalg.norm(axis) + 1e-8
    theta = np.pi * np.random.rand()
    euler = axis * theta
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation

def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> np.ndarray:
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0  # (0, rot_mag)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform

def random_sample_transform_v2(yaw_magnitude:float, pitch_magnitude:float,  roll_magnitude:float, translation_magnitude:float) -> np.ndarray:
    euler = np.random.rand(3)*[yaw_magnitude, pitch_magnitude, roll_magnitude]* np.pi / 180.0
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform

# Sampling methods


def random_sample_keypoints(
    points: np.ndarray,
    feats: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.random.choice(num_points, num_keypoints, replace=False)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_keypoints_with_scores(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.argsort(-scores)[:num_keypoints]
        points = points[indices]
        feats = feats[indices]
    return points, feats


def random_sample_keypoints_with_scores(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.arange(num_points)
        probs = scores / np.sum(scores)
        indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_keypoints_with_nms(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        radius2 = radius ** 2
        masks = np.ones(num_points, dtype=np.bool)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_points = points[sorted_indices]
        sorted_feats = feats[sorted_indices]
        indices = []
        for i in range(num_points):
            if masks[i]:
                indices.append(i)
                if len(indices) == num_keypoints:
                    break
                if i + 1 < num_points:
                    current_masks = np.sum((sorted_points[i + 1 :] - sorted_points[i]) ** 2, axis=1) < radius2
                    masks[i + 1 :] = masks[i + 1 :] & ~current_masks
        points = sorted_points[indices]
        feats = sorted_feats[indices]
    return points, feats


def random_sample_keypoints_with_nms(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        radius2 = radius ** 2
        masks = np.ones(num_points, dtype=np.bool)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_points = points[sorted_indices]
        sorted_feats = feats[sorted_indices]
        indices = []
        for i in range(num_points):
            if masks[i]:
                indices.append(i)
                if i + 1 < num_points:
                    current_masks = np.sum((sorted_points[i + 1 :] - sorted_points[i]) ** 2, axis=1) < radius2
                    masks[i + 1 :] = masks[i + 1 :] & ~current_masks
        indices = np.array(indices)
        if len(indices) > num_keypoints:
            sorted_scores = scores[sorted_indices]
            scores = sorted_scores[indices]
            probs = scores / np.sum(scores)
            indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        points = sorted_points[indices]
        feats = sorted_feats[indices]
    return points, feats


# depth image utilities


def convert_depth_mat_to_points(
    depth_mat: np.ndarray, intrinsics: np.ndarray, scaling_factor: float = 1000.0, distance_limit: float = 6.0
):
    r"""Convert depth image to point cloud.

    Args:
        depth_mat (array): (H, W)
        intrinsics (array): (3, 3)
        scaling_factor (float=1000.)

    Returns:
        points (array): (N, 3)
    """
    focal_x = intrinsics[0, 0]
    focal_y = intrinsics[1, 1]
    center_x = intrinsics[0, 2]
    center_y = intrinsics[1, 2]
    height, width = depth_mat.shape
    coords = np.arange(height * width)
    u = coords % width
    v = coords / width
    depth = depth_mat.flatten()
    z = depth / scaling_factor
    z[z > distance_limit] = 0.0
    x = (u - center_x) * z / focal_x
    y = (v - center_y) * z / focal_y
    points = np.stack([x, y, z], axis=1)
    points = points[depth > 0]
    return points

def ssc_to_homo(ssc):
    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation

    sr = np.sin(np.pi/180.0 * ssc[3])
    cr = np.cos(np.pi/180.0 * ssc[3])

    sp = np.sin(np.pi/180.0 * ssc[4])
    cp = np.cos(np.pi/180.0 * ssc[4])

    sh = np.sin(np.pi/180.0 * ssc[5])
    ch = np.cos(np.pi/180.0 * ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H