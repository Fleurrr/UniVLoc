import cv2
import numpy as np

import pdb

def create_homogeneous_matrix(K, R, t):
    """
    Create a homogeneous transformation matrix.
    
    Args:
    - K: Intrinsic camera matrix
    - R: Rotation matrix
    - t: Translation vector
    
    Returns:
    - H: Homogeneous transformation matrix
    """
    H = np.zeros((4, 4))
    H[:3, :3] = np.dot(K, R)
    H[:3, 3] = np.dot(K, t)
    H[3, 3] = 1
    return H

def warp_image(img, H, output_shape):
    """
    Warp an image using a given homography matrix.
    
    Args:
    - img: Input image
    - H: Homography matrix
    - output_shape: Shape of the output image
    
    Returns:
    - Warped image
    """
    return cv2.warpPerspective(img, H, output_shape)

def create_panorama(images, Ks, Rs, ts):
    """
    Stitch four images together.
    
    Args:
    - images: List of input images
    - Ks: List of intrinsic camera matrices
    - Rs: List of rotation matrices
    - ts: List of translation vectors
    
    Returns:
    - Stitched image
    """
    
    # Define the output shape of the stitched image
    output_shape = (800, 800)  # Example shape, adjust as needed
    
    # Create homography matrices
    Hs = [create_homogeneous_matrix(K, R, t) for K, R, t in zip(Ks, Rs, ts)]
    
    # Warp images
    warped_images = [warp_image(img, H, output_shape) for img, H in zip(images, Hs)]
    
    # Create a blank image to store the stitched image
    stitched_image = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)
    
    # Blend the warped images
    for warped_img in warped_images:
        stitched_image = cv2.addWeighted(stitched_image, 1, warped_img, 0.5, 0)
    
    return stitched_image

def raw_data_to_panorama(imgs, Ps, Ks):
    imgs = imgs.squeeze().permute(0, 2, 3, 1).detach().numpy()
    Ps = Ps.squeeze().detach().numpy()
    Ks = Ks.squeeze().detach().numpy()
    images = [imgs[0], imgs[1], imgs[2], imgs[3]]
    ks = [Ks[0], Ks[1], Ks[2], Ks[3]]
    rs = [Ps[0, :3, :3], Ps[1, :3, :3], Ps[2, :3, :3], Ps[3, :3, :3]]
    ts = [Ps[0, 3, :3], Ps[1, 3, :3], Ps[2, 3, :3], Ps[3, 3, :3]]
    panorama = create_panorama(images, ks, rs, ts)
    return panorama

if __name__ == '__main__':
    # Load input images
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    img3 = cv2.imread('image3.jpg')
    img4 = cv2.imread('image4.jpg')
    
    images = [img1, img2, img3, img4]
    
    # Intrinsic camera parameters (focal length, principal point)
    K1 = np.array([[1000, 0, 320],
                   [0, 1000, 240],
                   [0, 0, 1]])
    
    K2 = np.array([[1000, 0, 320],
                   [0, 1000, 240],
                   [0, 0, 1]])
    
    K3 = np.array([[1000, 0, 320],
                   [0, 1000, 240],
                   [0, 0, 1]])
    
    K4 = np.array([[1000, 0, 320],
                   [0, 1000, 240],
                   [0, 0, 1]])
    
    Ks = [K1, K2, K3, K4]
    
    # Rotation matrices
    R1 = np.eye(3)
    R2 = np.eye(3)
    R3 = np.eye(3)
    R4 = np.eye(3)
    
    Rs = [R1, R2, R3, R4]
    
    # Translation vectors
    t1 = np.array([0, 0, 1])
    t2 = np.array([0, 0, 1])
    t3 = np.array([0, 0, 1])
    t4 = np.array([0, 0, 1])
    
    ts = [t1, t2, t3, t4]
    
    # Stitch images
    result = main(images, Ks, Rs, ts)
    
    # Display the result
    cv2.imshow('Stitched Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()