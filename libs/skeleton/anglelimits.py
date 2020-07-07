"""
Utility functions for the hierarchical human representation. 
A Python implementation for pose-conditioned joint angle limits is also included.
Reference: "Pose-Conditioned Joint Angle Limits for 3D Human Pose Reconstruction"
"""
import logging
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#=============================================================================#
# Load the joint angle constraints
# These files are directly converted from .mat to .npy    
# The MATLAB implementation of the CVPR 15 paper has detailed documentation.
root = "../resources/constraints"
logging.info("Loading files from " + root)
model_path = os.path.join(root, "jointAngleModel_v2.npy")
joint_angle_limits = np.load(model_path, allow_pickle=True).item()
angle_spread = joint_angle_limits['angleSprd']
# separation plane for conditional joint angle
sepPlane = joint_angle_limits['sepPlane']
E2 = joint_angle_limits['E2']
bounds = joint_angle_limits['bounds']
# static pose and parameters used in coordinate transformation
static_pose_path = os.path.join(root, "staticPose.npy")
static_pose = np.load(static_pose_path, allow_pickle=True).item()
di = static_pose['di']
a = static_pose['a'].reshape(3)
# load the pre-computed conditinal distribution 
con_dis_path = os.path.join(root, "conditional_dis.npy")
con_dis = np.load(con_dis_path, allow_pickle=True).item()
#=============================================================================#
# joint names of the CVPR 15 paper
PRIOR_NAMES = ['back-bone', 
               'R-shldr', 
               'R-Uarm', 
               'R-Larm', 
               'L-shldr', 
               'L-Uarm', 
               'L-Larm', 
               'head', 
               'R-hip', 
               'R-Uleg', 
               'R-Lleg', 
               'R-feet', 
               'L-hip',
               'L-Uleg', 
               'L-Lleg', 
               'L-feet'
               ]
# Human 3.6M joint names are slightly different from the above
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'
# correspondence of the joints 
# (key, value) -> (index in prior_names, index in H36M names)
correspondence = {0:12, 1:13, 2:25, 3:26, 4:27, 5:17, 6:18, 7:19, 8:15, 9:1,
                  10:2, 11:3, 13:6, 14:7, 15:8}
# number of bone vectors attached to a torso
num_of_bones = 9        
# descretization of spherical coordinates
# bin edges for theta
theta_edges = np.arange(0.5, 122, 1)# theta values: 1 to 121 (integer)
# bin edges for phi
phi_edges = np.arange(0.5, 62, 1) # phi values: 1 to 61
# color map used for visualization
cmap = plt.cm.RdYlBu
# indices used for computing bone vectors for non-torso bones
nt_parent_indices = [13, 17, 18, 25, 26, 6, 7, 1, 2, 13]
nt_child_indices = [15, 18, 19, 26, 27, 7, 8, 2, 3, 14]
# map from bone index to the parent's di index
# TODO
di_indices = {2:5, 4:2, 6:13, 8:9}
# map from angle index to record index
record_indices = {0:4, 1:2, 3:0, 5:8, 7:5, 2:3, 4:1, 6:9, 8:6}
# name for the bone vectors
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
#=============================================================================#
def is_valid_local(skeleton_local, return_ang = False):
    """
    Check if the limbs represented in local coordinate system are valid or not.
    """
    valid_vec = np.ones((num_of_bones), dtype=np.bool)
    angles = to_spherical(skeleton_local)
    angles[:,1:] *= 180/np.pi
    # convert to valid range and discretize
    # theta: -180~180 degrees discretized into 120 bins
    # phi: -90~90 degrees discretized into 60 bins
    angles[:, 1] = np.floor((angles[:, 1]+180)/3 + 1)
    angles[:, 2] = np.floor((angles[:, 2]+90)/3 + 1)
    # go through each bone and check the angle-limits
    for angle_idx in range(len(angles)):
        angle = angles[angle_idx]
        record_idx = record_indices[angle_idx]        
        theta, phi = int(angle[1]), int(angle[2])
        if angle_idx in [0, 1, 3, 5, 7]:       
            test_value = angle_spread[0, record_idx][theta-1, phi-1]
            if test_value == 0:
                valid_vec[angle_idx] == False
        else:
            angle_parent = angles[angle_idx - 1]
            theta_p, phi_p = int(angle_parent[1]), int(angle_parent[2])
            vector = normalize(sepPlane[0, record_idx][theta_p-1, phi_p-1])
            for value in vector:
                if np.isnan(value):
                    valid_vec[angle_idx] = False
                    continue
            if np.dot(np.hstack([skeleton_local[angle_idx], 1]), vector) > 0:
                valid_vec[angle_idx] = False
            else:
                e1 = vector[:-1]
                e2 = E2[0, record_idx][theta_p-1, phi_p-1]
                T = gram_schmidt_columns(np.hstack([e1.reshape(3,1),
                                                    e2.reshape(3,1),
                                                    np.cross(e1,e2).reshape(3,1)]))
                bnd = bounds[0, record_idx][theta_p-1, phi_p-1]
                u = (T[:, 1:]).T @ skeleton_local[angle_idx]
                if u[0] < bnd[0] or u[0] > bnd[1] or u[1] < bnd[2] or u[1] > bnd[3]:
                    valid_vec[angle_idx] = False    
    if return_ang:
        return valid_vec, angles
    else:
        return valid_vec

def is_valid(skeleton, return_ang = False, camera = False):
    """
    args:
        skeleton: input skeleton of shape [num_joints, 3] use the annotation 
        of Human 3.6M dataset
    return:
        valid_vec: boolean vector specifying the validity for each bone. 
        return 0 for invalid bones.
        camera: relative orientation of camera and human
    """
    
    skeleton = skeleton.reshape(len(H36M_NAMES), -1)
    # the ordering of coordinate used by the Prior was x,z and y
    skeleton = skeleton[:, [0,2,1]]
    # convert bone vectors into local coordinate
    skeleton_local = to_local(skeleton)
    ret = is_valid_local(skeleton_local, return_ang=return_ang)
    if return_ang:
        return ret[0], ret[1]
    else:
        return ret

def normalize(vector):
    """
    Normalize a vector.
    """
    return vector/np.linalg.norm(vector)

def to_spherical(xyz):
    """
    Convert from Cartisian coordinate to spherical coordinate
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
    Convert from spherical coordinate to Cartisian coordinate
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

def test_coordinate_conversion():
    # theta: [-pi, pi] reference
    xyz = np.random.rand(1, 3)*2 - 1
    rthetaphi = to_spherical(xyz)
    xyz2 = to_xyz(rthetaphi)
    print('maximum error:', np.max(np.abs(xyz - xyz2)))
    # theta: [0, 2*pi] reference
    xyz = np.random.rand(1, 3)*2 - 1
    rthetaphi = to_spherical(xyz)
    indices = rthetaphi[:,1] < 0
    rthetaphi[:,1][indices] += 2*np.pi
    xyz2 = to_xyz(rthetaphi)
    print('maximum error:', np.max(np.abs(xyz - xyz2)))    
    return

def gram_schmidt_columns(X):
    """
    Apply Gram-Schmidt orthogonalization to obtain basis vectors.
    """
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

def direction_check(system, v1, v2, v3):
    if system[:,0].dot(v1) <0:
        system[:,0] *= -1
    if system[:,1].dot(v2) <0:
        system[:,1] *= -1
    if system[:,2].dot(v3) <0:
        system[:,2] *= -1        
    return system

def get_normal(x1, a, x):
    """
    Get normal vector.
    """
    nth = 1e-4
    # x and a are parallel
    if np.linalg.norm(x - a) < nth or np.linalg.norm(x + a) < nth:   
        n = np.cross(x, x1)
        flag = True
    else:
        n = np.cross(a, x)
        flag = False
    return normalize(n), flag

def get_basis1(skeleton):
    """
    Compute local coordinate system from 3D joint positions.
    This system is used for upper-limbs.
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

def to_local(skeleton):
    """
    Represent the bone vectors in the local coordinate systems.
    """
    v1, v2, v3 = get_basis1(skeleton)
    # compute the vector from the left hip to the right hip
    left_hip = skeleton[6]
    right_hip = skeleton[1]
    v4 = normalize(right_hip - left_hip)
    # v5 is the cross product of v4 and v2 (front-facing vector for lower-body)
    v5 = normalize(np.cross(v4, v2))
    # compute orthogonal coordinate systems using GramSchmidt
    # for upper body, we use v1, v2 and v3
    system1 = gram_schmidt_columns(np.hstack([v1.reshape(3,1), 
                                              v2.reshape(3,1), 
                                              v3.reshape(3,1)]))
    # make sure the directions rougly align
    #system1 = direction_check(system1, v1, v2, v3)
    # for lower body, we use v4, v2 and v5
    system2 = gram_schmidt_columns(np.hstack([v4.reshape(3,1), 
                                              v2.reshape(3,1), 
                                              v5.reshape(3,1)]))
    #system2 = direction_check(system2, v4, v2, v5)

    bones = skeleton[nt_parent_indices, :] - skeleton[nt_child_indices, :]    
    # convert bone vector to local coordinate system
    bones_local = np.zeros(bones.shape, dtype=bones.dtype)
    for bone_idx in range(len(bones)):
        # only compute bone vectors for non-torsos
        # the order of the non-torso bone vector is: 
        # bone vector1: thorax to head top
        # bone vector2: left shoulder to left elbow
        # bone vector3: left elbow to left wrist
        # bone vector4: right shoulder to right elbow
        # bone vector5: right elbow to right wrist
        # bone vector6: left hip to left knee
        # bone vector7: left knee to left ankle
        # bone vector8: right hip to right knee
        # bone vector9: right knee to right ankle        
        bone = normalize(bones[bone_idx])
        if bone_idx in [0, 1, 3, 5, 7]:
            # bones that are directly connected to the torso
            if bone_idx in [0, 1, 3]:
                # upper body
                bones_local[bone_idx] = system1.T @ bone
            else:
                # lower body
                bones_local[bone_idx] = system2.T @ bone
        else:
            if bone_idx in [2, 4]:
                parent_R = system1
            else:
                parent_R = system2
            # parent bone index is smaller than 1
            vector_u = normalize(bones[bone_idx - 1])
            di_index = di_indices[bone_idx]
            vector_v, flag = get_normal(parent_R@di[:, di_index],
                                        parent_R@a,
                                        vector_u
                                        )
            vector_w = np.cross(vector_u, vector_v)
            local_system = gram_schmidt_columns(np.hstack([vector_u.reshape(3,1),
                                                           vector_v.reshape(3,1),
                                                           vector_w.reshape(3,1)]
                                                          )
                                                )
            bones_local[bone_idx] = local_system.T @ bone
    return bones_local

def to_global(skeleton, bones_local, cache=False):
    """
    Convert local coordinate back into global coordinate system.
    cache: return intermeadiate results
    """
    return_value = {}
    v1, v2, v3 = get_basis1(skeleton)
    # compute the vector from the left hip to the right hip
    left_hip = skeleton[6]
    right_hip = skeleton[1]
    v4 = normalize(right_hip - left_hip)
    # v5 is the cross product of v4 and v2 (front-facing vector for lower-body)
    v5 = normalize(np.cross(v4, v2))
    # compute orthogonal coordinate systems using GramSchmidt
    # for upper body, we use v1, v2 and v3
    system1 = gram_schmidt_columns(np.hstack([v1.reshape(3,1), 
                                              v2.reshape(3,1), 
                                              v3.reshape(3,1)]))
    # make sure the directions rougly align
    #system1 = direction_check(system1, v1, v2, v3)
    # for lower body, we use v4, v2 and v5
    system2 = gram_schmidt_columns(np.hstack([v4.reshape(3,1), 
                                              v2.reshape(3,1), 
                                              v5.reshape(3,1)]))
    #system2 = direction_check(system2, v4, v2, v5)
    if cache:
        return_value['cache'] = [system1, system2]
        return_value['bl'] = bones_local
        
    bones_global = np.zeros(bones_local.shape)   
    # convert bone vector to local coordinate system
    for bone_idx in [0,1,3,5,7,2,4,6,8]:
        # the indices follow the order from torso to limbs
        # only compute bone vectors for non-torsos      
        bone = normalize(bones_local[bone_idx])
        if bone_idx in [0, 1, 3, 5, 7]:
            # bones that are directly connected to the torso
            if bone_idx in [0, 1, 3]:
                # upper body
                # this is the inverse transformation compared to the to_local 
                # function
                bones_global[bone_idx] = system1 @ bone
            else:
                # lower body
                bones_global[bone_idx] = system2 @ bone
        else:
            if bone_idx in [2, 4]:
                parent_R = system1
            else:
                parent_R = system2
            # parent bone index is smaller than 1
            vector_u = normalize(bones_global[bone_idx - 1])
            di_index = di_indices[bone_idx]
            vector_v, flag = get_normal(parent_R@di[:, di_index],
                                  parent_R@a,
                                  vector_u)
            vector_w = np.cross(vector_u, vector_v)
            local_system = gram_schmidt_columns(np.hstack([vector_u.reshape(3,1), 
                                              vector_v.reshape(3,1), 
                                              vector_w.reshape(3,1)]))
            if cache:
                return_value['cache'].append(local_system)
            bones_global[bone_idx] = local_system @ bone
    return_value['bg'] = bones_global
    return return_value

def test_global_local_conversion():
    """
    test for global and lobal coordinate conversion
    """
    path='Your3DSkeleton.npy'
    index = 0
    pose = np.load(path, allow_pickle=True)[index] 
    pose = pose.reshape(32, -1)    
    global_bones = pose[nt_parent_indices, :] - pose[nt_child_indices, :]    
    for bone_idx in range(len(global_bones)):    
        global_bones[bone_idx] = normalize(global_bones[bone_idx])    
    local_c = to_local(pose)
    global_c = to_global(pose, local_c)['bg']
    maximum_error = np.max(np.abs(global_bones - global_c))
    print('maximum error', maximum_error)
    return maximum_error

def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
               gt=False,pred=False,inv_z=False): # blue, orange

    vals = np.reshape( channels, (32, -1) )

    I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
    J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
    LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        if gt:
            ax.plot(x,y, z,  lw=2, c='k')
        
        elif pred:
            ax.plot(x,y, z,  lw=2, c='r')

        else:
            ax.plot(x,y, z,  lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 750 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

#    if add_labels:
#        ax.set_xlabel("x")
#        ax.set_ylabel("z")
#        ax.set_zlabel("y")

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
    ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)
    if inv_z:
        ax.invert_zaxis()  
        
def get_histogram2d(angles, validmap=None, smooth=True):
    """
    Obtain a 2D histogram for discretized joint angles
    """
    H, xedges, yedges = np.histogram2d(angles[:,0], 
                                       angles[:,1], 
                                       bins=(theta_edges, phi_edges)
                                       )    
    if validmap is not None:
        # rule out outliers
        mask = validmap != 0
        H = H * mask
    H = H.reshape(-1)
    H = H/H.sum()
    return H

def sample_from_histogram(histogram, 
                          x=np.arange(1,122,1), 
                          y=np.arange(1,62,1), 
                          total=1000, 
                          add_noise=False, 
                          bin_size=3
                          ):
    """
    Sample from a pre-computed histogram.
    """
    assert histogram.shape[0] == len(x)
    assert histogram.shape[1] == len(y)
    # normalize the histogram
    histogram = histogram/histogram.sum()
    # multiply to get final counts
    histogram = histogram*total
    none_zeros = histogram!=0
    histogram = histogram.astype(np.int)     
    histogram[none_zeros] = histogram[none_zeros] + 1
    data = []
    for x_id in x:
        for y_id in y:
            counts = histogram[x_id-1, y_id-1]
            if counts!=0:
                temp = np.array([[x[x_id - 1], y[y_id - 1]]])
                data.append(np.repeat(temp, counts, axis=0))
    data = np.vstack(data)
    if add_noise:
        noise = np.random.rand(*(data.shape))*bin_size
        data = (data - 1)*bin_size + noise
    return data[:total,:]

def histogram_transform(histogram, gamma):
    """
    Transform a distribution with power function.
    """
    histogram = (histogram - histogram.min())/(histogram.max() - histogram.min())
    histogram = np.power(histogram, gamma)
    return histogram

#=============================================================================#
# Visualization utilities.
def smooth_histogram2d(data):
    """
    Smooth a 2D histogram with kernel density estimation.
    """
    from scipy.stats import kde    
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[1:122, 1:62]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # change the extent
    xi = xi*3 - 180
    yi = yi*3 - 180
    fig, axes = plt.subplots(ncols=1, nrows=3)
    # plot a density
    axes[0].set_title('Calculate Gaussian KDE')
    axes[0].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    # add shading
    axes[1].set_title('2D Density with shading')
    axes[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axes[1].set_aspect('equal') 
    axes[1].invert_yaxis()
    # contour
    axes[2].set_title('Contour')
    axes[2].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axes[2].contour(xi, yi, zi.reshape(xi.shape))    
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()
    return

def decorate_axis(ax, title=None):
    ax.set_xlabel('Theta: -180 to 180')
    ax.set_label('Phi:-90 to 90')
    if title is not None:
        ax.set_title(title)
    return

def adjust_figure(left = 0, 
                  right = 1, 
                  bottom = 0.01, 
                  top = 0.95,
                  wspace = 0, 
                  hspace = 0.4
                  ):  
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    return

def plot_relative_poses(skeletons, cameras=None):
    """
    Visualize the distribution of front vector in camera coordinate system.
    """
    # skeletons: 3D poses in world or camera coordinates
    # cameras: camera parameters
    vector_list = []
    if cameras is None:
        for pose_id in range(len(skeletons)):
            _, _, front_vector = get_basis1(skeletons[pose_id].reshape(32, -1))
            vector_list.append(front_vector)
    else:
        raise NotImplementedError
    vector_list = np.vstack(vector_list)
    spherical = to_spherical(vector_list)
    # convert to valid range and discretize
    spherical[:, 1:] *= 180/np.pi
    spherical[:, 1] = np.floor((spherical[:, 1]+180)/3 + 1)
    spherical[:, 2] = np.floor((spherical[:, 2]+90)/3 + 1)    
    H, xedges, yedges = np.histogram2d(spherical[:,1], spherical[:,2], 
                                       bins=(theta_edges, phi_edges))    
    plt.figure()
    ax = plt.subplot(111)
    plt.imshow(H.T, extent=[-180, 180, -90, 90], interpolation='bilinear')
    decorate_axis(ax, 'Relative camera pose in H36M')
    return vector_list

def plot_distribution(H_temp, 
                      angle_idx, 
                      dataset_name, 
                      gamma, 
                      save_path='../viz'
                      ):
    """
    Visualize distribution of limb orientation.
    """
    # angles: [n_samples, 2] in theta and phi order
    # plot the distribution of local joint angles and overlay valid regions
    plt.ioff()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    H = H_temp.copy()
    # normalize
    H = (H-H.min())/(H.max()-H.min())
    # perform "gamma correction"
    H = np.power(H, gamma)
    # normalize again
    H = (H-H.min())/(H.max()-H.min())
    H_return = H.copy().reshape(121,61)
    # map to color 
    H = cmap(H).reshape((121, 61, 4))
    H = [np.expand_dims(H[:,:,i].T, axis=2) for i in range(4)]
    H = np.concatenate(H, axis=2)
    if angle_idx in [0,1,3,5,7]:
        record_idx = record_indices[angle_idx]
        mask_temp = angle_spread[0, record_idx]
        mask = np.zeros((61, 121, 4))
        mask[:,:,3] = mask_temp.T
        mask[:,:,1] = mask_temp.T
        f = plt.figure(figsize=(5, 6))
        ax = plt.subplot(211)
        ax.imshow(H, extent=[-180, 180, -90, 90], interpolation='bilinear', alpha=1)    
        decorate_axis(ax, 'Distribution of ' + dataset_name + ' for bone: ' + bone_name[angle_idx+1]) 
        ax = plt.subplot(212)
        ax.imshow(mask, extent=[-180, 180, -90, 90],alpha=0.5)
        ax.imshow(H, extent=[-180, 180, -90, 90], interpolation='bilinear', alpha=0.5)
        decorate_axis(ax, 'Overlayed')
        plt.tight_layout()
    else:
        f = plt.figure()
        ax = plt.subplot(111)
        ax.imshow(H, extent=[-180, 180, -90, 90], interpolation='bilinear', alpha=1)    
        decorate_axis(ax, 'Distribution of ' + dataset_name + ' for bone: ' + bone_name[angle_idx+1])  
        plt.tight_layout()
    adjust_figure(left = 0.135, 
                  right = 0.95, 
                  bottom = 0.05, 
                  top = 1,
                  wspace = 0, 
                  hspace = 0
                  )
    save_name = dataset_name + bone_name[angle_idx+1] + '_gamma_' + str(gamma) + '.jpg'
    f.savefig(os.path.join(save_path, save_name))
    plt.close(f)
    return save_name, H_return

#=============================================================================#
# sampling utilities: sample 3D human skeleton from a pre-computed distribution
template = np.load(os.path.join(root, 'template.npy'), allow_pickle=True).reshape(32,-1)  
template_bones = template[nt_parent_indices, :] - template[nt_child_indices, :]  
template_bone_lengths = to_spherical(template_bones)[:, 0]
nt_parent_indices = [13, 17, 18, 25, 26, 6, 7, 1, 2]
nt_child_indices = [15, 18, 19, 26, 27, 7, 8, 2, 3]    
def get_skeleton(bones, pose, bone_length=template_bone_lengths):
    """
    Update the non-torso limb of a skeleton by specifying bone vectors.
    """
    new_pose = pose.copy()
    for bone_idx in [0,1,3,5,7,2,4,6,8]:
        new_pose[nt_child_indices[bone_idx]] = new_pose[nt_parent_indices[bone_idx]] \
        - bones[bone_idx]*bone_length[bone_idx]            
    return new_pose

def test_get_skeleton():
    pose = template
    nt_parent_indices = [13, 17, 18, 25, 26, 6, 7, 1, 2]
    nt_child_indices = [15, 18, 19, 26, 27, 7, 8, 2, 3]    
    global_bones = pose[nt_parent_indices, :] - pose[nt_child_indices, :]    
    for bone_idx in range(len(global_bones)):    
        global_bones[bone_idx] = normalize(global_bones[bone_idx])    
    local_c = to_local(pose)
    global_c = to_global(pose, local_c)['bg']
    new_pose = get_skeleton(global_c, pose)
    maximum_error = np.max(np.abs(new_pose - pose))
    print('maximum error', maximum_error)    
    return maximum_error

def grow_from_torso(poses, all_angles, cache=False):
    """
    Update the non-torso limb of a skeleton by specifying limb orientations.
    """
    new_poses = poses.copy()
    return_value = {}
    if cache:
        return_value['cache'] = []
        return_value['bl'] = []
    for pose_id in range(len(poses)):
        pose = poses[pose_id].reshape(32,-1)
        angles = all_angles[:, pose_id, :]
        # convert to spherical coordinate in radians
        spherical = np.ones((len(angles), 3))
        spherical[:, 1:] = angles/180*np.pi
        spherical[:, 1] -= np.pi
        spherical[:, 2] -= np.pi/2
        # convert to local cartisian coordinate
        local_xyz = to_xyz(spherical)
        # convert to global coordinate
        return_value_temp = to_global(pose, local_xyz, cache=cache)
        if cache:
            return_value['cache'].append(return_value_temp['cache'])
            return_value['bl'].append(return_value_temp['bl'])
        global_xyz = return_value_temp['bg']
        new_poses[pose_id] = get_skeleton(global_xyz, new_poses[pose_id].reshape(32,-1)).reshape(-1)
    return_value['np'] = new_poses
    return return_value

def test_grow_from_torso():
    poses = template.reshape(1, 96)
    _, angles = is_valid(poses.reshape(32, -1), return_ang=True)
    local_coordinate = to_local(poses.reshape(32, -1))
    angles = angles[:,1:].reshape(9, 1, 2)
    # to degrees
    angles *= 3
    return_dic = grow_from_torso(poses, angles, cache=True)
    new_poses = return_dic['np']
    bones_local = return_dic['bl']
    print(np.max(np.abs(local_coordinate - bones_local)))
    maximum_error = np.max(np.abs(new_poses - poses))
    print('maximum error1', maximum_error)       
    _, new_angles = is_valid(new_poses, return_ang=True)
    maximum_error = np.max(np.abs(angles[:,0,:]/3 - new_angles[:,1:]))
    print('maximum error2', maximum_error)    
    return 

def sample_lower_limbs(angles, bin_size=3):
    """
    Sample limb orientations for lower limbs.
    """
    # angles of shape [num_bones, sample_size, 2]
    for sample_id in range(angles.shape[1]):
        for angle_idx in [2, 4, 6, 8]:
            record_idx = record_indices[angle_idx]
            parent = angles[angle_idx - 1, sample_id, :]
            theta = np.floor(parent[0]/3)
            phi = np.floor(parent[1]/3) # convert to bins
            candidate_length = len(con_dis[record_idx][(theta, phi)])
            # change some boundary points
            if candidate_length == 0:
                keys = list(con_dis[record_idx].keys())
                while candidate_length == 0:
                    temp_idx = np.random.choice(len(keys), 1)
                    theta, phi = keys[temp_idx[0]]
                    candidate_length = len(con_dis[record_idx][(theta, phi)])
            chosen_idx = np.random.choice(candidate_length, 1)
            angles[angle_idx, sample_id, :] = con_dis[record_idx][(theta, phi)][chosen_idx]
    # convert to degrees with some noise
    angles[[2,4,6,8], :, :] = angles[[2,4,6,8], :, :]*bin_size 
    angles[[2,4,6,8], :, :] += np.random.rand(4, angles.shape[1], 2)*bin_size
    return angles

def sample_upper_limbs(angles, sample_num):
    """
    Sample limb orientation for upper limbs.
    """
    # sample torso limbs from the valid maps uniformly
    # angles of shape [num_bones, sample_size, 2]
    for angle_idx in [0, 1, 3, 5, 7]:
        valid_map = angle_spread[0, record_indices[angle_idx]]
        valid_map = valid_map.astype(np.float16)
        valid_map /= valid_map.sum()
        bone_angles = sample_from_histogram(valid_map, total=sample_num, 
                                add_noise=True)
        angles[angle_idx, :, :] = bone_angles
    return angles