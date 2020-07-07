import numpy as np 
import scipy.io as sio 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

bone_name = {
 1: 'thorax to head top',
 2: 'left shoulder to left elbow',
 3: 'left elbow to left wrist',
 4: 'right shoulder to right elbow',
 5: 'right elbow to right wrist',
 6: 'left hip to left knee',
 7: 'left knee to left ankle',
 8: 'right hip to right knee',
 9: 'right knee to right ankle'
}      

static_pose_path = "/media/nicholas/Database/Github/EvoSkeleton/resources/constraints/staticPose.npy"
static_pose = np.load(static_pose_path, allow_pickle=True).item()
di = static_pose['di']
a = static_pose['a'].reshape(3)

di_indices = {2:5, 4:2, 6:13, 8:9}
nt_parent_indices = [13, 17, 18, 25, 26, 6, 7, 1, 2]
nt_child_indices = [15, 18, 19, 26, 27, 7, 8, 2, 3]

def gram_schmidt_columns(X):
    B = np.zeros(X.shape)
    B[:, 0] = (1/np.linalg.norm(X[:, 0]))*X[:, 0]
    for i in range(1, 3):
        v = X[:, i]
        U = B[:, 0:i] # subspace basis which has already been orthonormalized
        pc = U.T @ v # orthogonal projection coefficients of v onto U
        p = U@pc
        v = v - p
        if np.linalg.norm(v) < 2e-16:
            # vectors are not linearly independent!
            raise ValueError
        else:
            v = normalize(v)
            B[:, i] = v
    return B

def normalize(vector):
    return vector/np.linalg.norm(vector)

def get_basis1(skeleton):
    """
    get local coordinate system
    """
    # compute the vector from the left shoulder to the right shoulder
    left_shoulder = skeleton[17]
    right_shoulder = skeleton[25]
    v1 = normalize(right_shoulder - left_shoulder)    
    # compute the backbone vector from the thorax to the spine 
    thorax = skeleton[13]
    spine = skeleton[12]
    v2 = normalize(spine - thorax)
    # v3 is the cross product of v1 and v2 (front-facing vector for upper-body)
    v3 = normalize(np.cross(v1, v2))    
    return v1, v2, v3

def get_normal(x1, a, x):
    nth = 1e-4
    # x and a are parallel
    if np.linalg.norm(x - a) < nth or np.linalg.norm(x + a) < nth:   
        n = np.cross(x, x1)
        flag = True
    else:
        n = np.cross(a, x)
        flag = False
    return normalize(n), flag

def to_spherical(xyz):
    """
    convert from Cartisian coordinate to spherical coordinate
    theta: [-pi, pi]
    phi: [-pi/2, pi/2]
    note that xyz should be float number
    """
    # return in r, phi, and theta (elevation angle from z axis down)
    return_value = np.zeros(xyz.shape, dtype=xyz.dtype)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    return_value[:,0] = np.sqrt(xy + xyz[:,2]**2) # r      
    return_value[:,1] = np.arctan2(xyz[:,1], xyz[:,0]) # theta
    return_value[:,2] = np.arctan2(xyz[:,2], np.sqrt(xy)) # phi
    
    return return_value

def to_xyz(rthetaphi):
    """
    convert from spherical coordinate to Cartisian coordinate
    theta: [0, 2*pi] or [-pi, pi]
    phi: [-pi/2, pi/2]
    """
    return_value = np.zeros(rthetaphi.shape, dtype=rthetaphi.dtype)
    sintheta = np.sin(rthetaphi[:,1])
    costheta = np.cos(rthetaphi[:,1])
    sinphi = np.sin(rthetaphi[:,2])
    cosphi = np.cos(rthetaphi[:,2])
    return_value[:,0] = rthetaphi[:,0]*costheta*cosphi # x
    return_value[:,1] = rthetaphi[:,0]*sintheta*cosphi # y
    return_value[:,2] = rthetaphi[:,0]*sinphi #z
    return return_value
