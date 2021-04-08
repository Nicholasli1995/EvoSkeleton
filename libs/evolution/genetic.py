"""
Utility functions for genetic evolution.
"""
import libs.dataset.h36m.cameras as cameras
from libs.skeleton.anglelimits import \
to_local, to_global, get_skeleton, to_spherical, \
nt_parent_indices, nt_child_indices, \
is_valid_local, is_valid

import matplotlib.pyplot as plt
import os
import logging

import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

root = "../resources/constraints"
# Joints in H3.6M -- data has 32 joints, but only 17 that move
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
total_joints_num = len(H36M_NAMES)

# this dictionary stores the parent indice for each joint
# key:value -> child joint index:its parent joint index
parent_idx = {1:0, 2:1, 3:2, 6:0, 7:6, 8:7, 12:0, 13:12, 14:13, 15:14, 17:13, 
              18:17, 19:18, 25:13, 26:25, 27:26
              }

# this dictionary stores the children indices for each parent joint
# key:value -> parent index: joint indices for its children as a list  
children_idx = {
        0: [1, 6],
        1: [2], 2: [3],
        6: [7], 7: [8],
        13: [14, 17, 25],
        14: [15], 17: [18], 18:[19],
        25: [26], 26:[27]
        }

# used roots for random selection
root_joints = [0, 1, 2, 6, 7, 13, 17, 18, 25, 26]
# names of the bone vectors attached on the human torso
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
# this dictionary stores the sub-tree rooted at each root joint
# key:value->root joint index:list of bone vector indices 
bone_indices = {0: [5, 6, 7, 8],
                1: [7, 8],
                2: [8],
                6: [5, 6],
                7: [6],
                13: [1, 2, 3, 4], # thorax
                17: [1, 2],
                18: [2],
                25: [3, 4],
                26: [4]
                }

# load template bone lengths that can be used during mutation
# you can prepare your own bone length templates to represent 
# subjects with different size
bl_templates = np.load(os.path.join(root, "bones.npy"), allow_pickle=True)

# pre-compute the sub-tree joint indices for each joint
subtree_indices = {}
def get_subtree(joint_idx, children_idx):
    if joint_idx not in children_idx:
        return None
    subtree = set()
    for child_idx in children_idx[joint_idx]:
        subtree.add(child_idx)
        offsprings = get_subtree(child_idx, children_idx)
        if offsprings is not None:
            subtree = subtree.union(offsprings)
    return subtree
for joint_idx in range(total_joints_num):
    if H36M_NAMES[joint_idx] != '':
        subtree_indices[joint_idx] = get_subtree(joint_idx, children_idx)

def swap_bones(bones_father, bones_mother, root_idx):
    swap_indices = bone_indices[root_idx]
    temp = bones_father.copy()
    bones_father[swap_indices] = bones_mother[swap_indices].copy()
    bones_mother[swap_indices] = temp[swap_indices].copy()
    del temp
    return bones_father, bones_mother, swap_indices

def get_bone_length(skeleton):
    """
    Compute limb length for a given skeleton.
    """
    bones = skeleton[nt_parent_indices, :] - skeleton[nt_child_indices, :]  
    bone_lengths = to_spherical(bones)[:, 0]    
    return bone_lengths

def get_h36m_bone_length(visualize=True):
    #1: F 5:F 6:M 7:F 8:M 9:M 11:M
    bl_dic = {1:[], 5:[], 6:[], 7:[], 8:[], 9:[], 11:[]}
    train_dic = np.load('../data/human3.6M/h36m/numpy/threeDPose_train.npy').item()
    test_dic = np.load('../data/human3.6M/h36m/numpy/threeDPose_test.npy').item()
    def process_dic(data_dic, bl_dic, candicate=50):
        for key in data_dic.keys():
            subject = key[0]
            indices = np.random.choice(len(data_dic[key]), candicate, replace=False)
            selected = data_dic[key][indices]
            for pose in selected:
                bl_dic[subject].append(get_bone_length(pose.reshape(32,3)))
        return
    process_dic(train_dic, bl_dic)
    process_dic(test_dic, bl_dic)
    for key in bl_dic:
        for array in bl_dic[key]:
            array = array.reshape(9,1)
        bl_dic[key] = np.vstack(bl_dic[key])    
    if visualize:
    # as can be observed, only bone length of idx 0 vary a lot. Others are almost fixed        
#        for key in bl_dic.keys():    
#            fig, axes = plt.subplots(3,3)
#            plt.title('Subject: '+str(key))
#            for row in range(3):
#                for col in range(3):
#                    axes[row][col].hist(bl_dic[key][:,3*row + col], bins=20)
        fig, axes = plt.subplots(3,3)
        all_lengths = np.vstack(list(bl_dic.values()))
        for row in range(3):
            for col in range(3):
                axes[row][col].hist(all_lengths[:,3*row + col])        
    return bl_dic

def get_random_rotation(sigma=60.):
    angle = np.random.normal(scale=sigma)
    axis_idx = np.random.choice(3, 1)
    if axis_idx == 0:
        r = R.from_euler('xyz', [angle, 0., 0.], degrees=True)
    elif axis_idx == 1:
        r = R.from_euler('xyz', [0., angle, 0.], degrees=True)
    else:
        r = R.from_euler('xyz', [0., 0., angle], degrees=True)    
    return r

def rotate_bone_random(bone, sigma=10.):
    r = get_random_rotation(sigma)
    bone_rot = r.as_dcm() @ bone.reshape(3,1)
    return bone_rot.reshape(3)

def rotate_pose_random(pose=None, sigma=60.):
    # pose shape: [n_joints, 3]
    if pose is None:
        result = None
    else:
        r = get_random_rotation()
        pose = pose.reshape(32, 3)
        # rotate around hip
        hip = pose[0].reshape(1, 3)
        relative_pose = pose - hip
        rotated = r.as_dcm() @ relative_pose.T
        result = rotated.T + hip
    return result

def re_order(skeleton):
    # the ordering of coordinate used by the Prior was x,z and y
    return skeleton[:, [0,2,1]]

def set_z(pose, target):
    if pose is None:
        return None
    original_shape = pose.shape
    pose = pose.reshape(32, 3)
    min_val = pose[:, 2].min()
    pose[:, 2] -= min_val - target
    return pose.reshape(original_shape)
    
def modify_pose(skeleton, local_bones, bone_length, ro=False):
    # get a new pose by modify an existing pose with input local bone vectors
    # and bone lengths
    new_bones = to_global(skeleton, local_bones)['bg']
    new_pose = get_skeleton(new_bones, skeleton, bone_length=bone_length)
    if ro:
        new_pose = re_order(new_pose)
    return new_pose.reshape(-1)

def exploration(father, mother, opt, post_processing=True):
    """
    Produce novel data by exploring the data space with evolutionary operators.
    cross over operator in the local coordinate system
    mutation: perturb the local joint angle
    """
    # get local coordinate for each bone vector
    father = re_order(father.reshape(total_joints_num, -1))
    father_bone_length = get_bone_length(father)
    mother = re_order(mother.reshape(total_joints_num, -1))
    mother_bone_length = get_bone_length(mother)
    bones_father = to_local(father)
    bones_mother = to_local(mother)    
    if opt.CV:
        # crossover: exchange random sub-trees of two kinematic trees
        root_idx = np.random.randint(0, len(root_joints))
        root_selected = root_joints[root_idx]
        bones_father, bones_mother, indices = swap_bones(bones_father, 
                                                         bones_mother, 
                                                         root_selected)
    if opt.M:
        # local mutation: apply random rotation to local limb
        for bone_idx in indices:
            if np.random.rand() <= opt.MRL:
                bones_father[bone_idx] = rotate_bone_random(bones_father[bone_idx], sigma=opt.SDL)
                bones_mother[bone_idx] = rotate_bone_random(bones_mother[bone_idx], sigma=opt.SDL)
                
    son_pose, daughter_pose = None, None
    if opt.C:
        # apply joint angle constraint as the fitness function
        valid_vec_fa = is_valid_local(bones_father)
        valid_vec_mo = is_valid_local(bones_mother)
    if not opt.C or valid_vec_fa.sum() >= opt.Th:
        son_pose = modify_pose(father, bones_father, mother_bone_length, ro=True)
    if not opt.C or valid_vec_mo.sum() >= opt.Th:
        daughter_pose = modify_pose(mother, bones_mother, father_bone_length, ro=True)
    if opt.M:
        # global mutation: rotate the whole 3D skeleton
        if np.random.rand() <= opt.MRG:
            son_pose = rotate_pose_random(son_pose, sigma=opt.SDG)
        if np.random.rand() <= opt.MRG:
            daughter_pose = rotate_pose_random(daughter_pose, sigma=opt.SDG)
    if post_processing:
        # move the poses to the ground plane
        set_z(son_pose, np.random.normal(loc=20.0, scale=3.0))
        set_z(daughter_pose, np.random.normal(loc=20.0, scale=3.0))
    if opt.DE:
        valid_vec_fa = is_valid_local(bones_father)
        valid_vec_mo = is_valid_local(bones_mother)        
        # re_order: order back to x, y, z
        son_pose = modify_pose(father, bones_father, mother_bone_length, ro=True)
        daughter_pose = modify_pose(mother, bones_mother, father_bone_length, ro=True)
#        valid_vec_son = is_valid(son_pose)
#        valid_vec_dau = is_valid(daughter_pose)             
    if opt.DE and opt.V:
        plt.figure()
        ax1 = plt.subplot(1,4,1, projection='3d')
        plt.title('father')
        show3Dpose(re_order(father), ax1, add_labels=False, plot_dot=True)
        ax2 = plt.subplot(1,4,2, projection='3d')
        plt.title('mother')
        show3Dpose(re_order(mother), ax2, add_labels=False, plot_dot=True)
        ax3 = plt.subplot(1,4,3, projection='3d')
        plt.title('son: ' + str(valid_vec_fa.sum()))
        show3Dpose(son_pose, ax3, add_labels=False, plot_dot=True)
        ax4 = plt.subplot(1,4,4, projection='3d')
        plt.title('daughter: ' + str(valid_vec_mo.sum()))
        show3Dpose(daughter_pose, ax4, add_labels=False, plot_dot=True)
        plt.tight_layout() 
    return son_pose, daughter_pose

def show3Dpose(channels, 
               ax, 
               lcolor="#3498db", 
               rcolor="#e74c3c", 
               add_labels=True,
               gt=False,
               pred=False,
               plot_dot=False
               ): # blue, orange
  """
  Visualize a 3d skeleton

  Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  if channels.shape[0] == 96:
      vals = np.reshape( channels, (32, -1) )
  else:
      vals = channels
  I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
  J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  dim_use_3d = [3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 
                26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 
                54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83]
  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    if gt:
        ax.plot(x,y, z,  lw=4, c='k')
#        ax.plot(x,y, z,  lw=2, c='k')
        
    elif pred:
        ax.plot(x,z, -y,  lw=4, c='r')
#        ax.plot(x,y, z,  lw=2, c='r')

    else:
#        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
        ax.plot(x,y, z,  lw=4, c=lcolor if LR[i] else rcolor)
  if plot_dot:
      joints = channels.reshape(96)
      joints = joints[dim_use_3d].reshape(16,3)
      ax.scatter(joints[:,0], joints[:,1], joints[:,2], c='k', marker='o')
  RADIUS = 750 # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

  ax.set_aspect('equal')
#  ax.set_xticks([])
#  ax.set_yticks([])
#  ax.set_zticks([])

#  ax.get_xaxis().set_ticklabels([])
#  ax.get_yaxis().set_ticklabels([])
#  ax.set_zticklabels([])
  # Get rid of the panes (actually, make them white)
#  white = (1.0, 1.0, 1.0, 0.0)
#  ax.w_xaxis.set_pane_color(white)
#  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
#  ax.w_xaxis.line.set_color(white)
#  ax.w_yaxis.line.set_color(white)
#  ax.w_zaxis.line.set_color(white)
  ax.view_init(10, -60)

def choose_best(population, fraction = 0.02, method='random'):
    """
    Choose the best candidates to produce descendents.
    """
    if method == 'random':
        # this is a simple implementation by random sampling
        num_total = len(population)
        num_to_choose = int(fraction*num_total)
        chosen_indices = np.random.choice(num_total, num_to_choose*2, replace=False)
        father_indices = chosen_indices[:num_to_choose]
        mother_indices = chosen_indices[num_to_choose:]
    else:
        raise NotImplementedError
    return father_indices, mother_indices

def project_to_cameras(poses, cams):
    """
    Project 3d poses using camera parameters
    input: 
        3D poses: [n_pose, pose length]
        cams: list of camera parameters
    return: list of 2D projections for each camera
    cams:
    """
    p_2d = []
    for cam in cams:
        R, T, f, c, k, p, name = cam
        pts2d, _, _, _, _ = cameras.project_point_radial(np.reshape(poses, [-1, 3]), R, T, f, c, k, p )
        p_2d.append(np.reshape( pts2d, [-1, len(H36M_NAMES)*2]))
    return p_2d

def transform_world_to_camera(poses, cams):
    """
    Project 3d poses from world coordinate to camera coordinate system
    return: list of 3D poses in camera coordinate systems
    """
    p_3d_cam = []
    for cam in cams:
        R, T, f, c, k, p, name = cam
        camera_coord = cameras.world_to_camera_frame( np.reshape(poses, [-1, 3]), R, T)
        camera_coord = np.reshape( camera_coord, [-1, len(H36M_NAMES)*3] )
        p_3d_cam.append(camera_coord)
    return p_3d_cam

def normalize(data, mean=None, std=None):
    if mean is not None and std is not None:
        pass
    elif mean is None and std is None:
        mean = np.mean(data, axis=0).reshape(1, data.shape[1])
        std = np.std(data, axis=0).reshape(1, data.shape[1])
    else:
        raise ValueError
    return (data-mean)/std

def unnormalize(data, mean, std):
    return (data*std) + mean

def postprocess_3d( poses):
    return poses - np.tile( poses[:,:3], [1, len(H36M_NAMES)] ) 

def calc_errors(pred_poses, gt_poses, protocol='mpjpe'):
    # error after a regid alignment, corresponding to protocol #2 in the paper
    # Compute Euclidean distance error per joint
    sqerr = (pred_poses - gt_poses)**2 # Squared error between prediction and expected output
    sqerr = sqerr.reshape(len(sqerr), -1, 3)
    sqerr = np.sqrt(sqerr.sum(axis=2))
    if protocol == 'mpjpe':
        ret = sqerr.mean(axis=1)
        ret = ret.reshape(len(ret), 1)
    else:
        raise NotImplementedError
    return ret

def to_numpy(tensor):
    return tensor.data.cpu().numpy()

def get_prediction(cascade, data):
    data = torch.from_numpy(data.astype(np.float32))
    if torch.cuda.is_available():
        data = data.cuda()
    # forward pass to get prediction for the first stage
    prediction = cascade[0](data)
    # prediction for later stages
    for stage_idx in range(1, len(cascade)):
        prediction += cascade[stage_idx](data)    
    return prediction

def get_score(model_file, candidates):
    """
    Obtain model inference errors for the candidates.
    """
    cams = model_file['cams']
    stats = model_file['stats']
    model = model_file['model']
    if torch.cuda.is_available():
        model = model.cuda()
    # project to 2D keypoints
    p_2d = project_to_cameras(candidates, cams)
    # convert to camera coordinate
    p_3d_cam = transform_world_to_camera(candidates, cams)
    # re-center relative to the hip
    for idx in range(len(p_3d_cam)):
        p_3d_cam[idx] = postprocess_3d(p_3d_cam[idx])
    # normalize the inputs
    dim_use_2d = stats['dim_use_2d']
    dim_use_3d = stats['dim_use_3d']
    mean_2d = stats['mean_2d'][dim_use_2d]
    std_2d = stats['std_2d'][dim_use_2d]
    mean_3d = stats['mean_3d'][dim_use_3d]
    std_3d = stats['std_3d'][dim_use_3d]
    for idx in range(len(p_2d)):
        p_2d[idx] = p_2d[idx][:, dim_use_2d]
        p_2d[idx] = normalize(p_2d[idx], mean_2d, std_2d)
    # get output and calculate errors
    output = []
    for idx in range(len(p_2d)):
        prediction = to_numpy(get_prediction(model, p_2d[idx]))
        # unnormalize the prediction
        prediction = unnormalize(prediction, mean_3d, std_3d)
        output.append(prediction)
    errors = []
    for idx in range(len(output)):
        gt_poses = p_3d_cam[idx][:, dim_use_3d]
        errors.append(calc_errors(output[idx], gt_poses))
    all_errors = np.concatenate(errors, axis = 1)
    # mean error for all the cameras
    mean_errors = all_errors.mean(axis = 1)
    return mean_errors

def active_select(model_file, candidates, ratio):
    """
    Actively select candidates that cause the model to fail.
    """
    scores = get_score(model_file, candidates)
    indices = np.argsort(scores) # error from low to high
    indices = indices[-int(ratio*len(candidates)):]
    mean_error = scores[indices].mean()
    return candidates[indices], mean_error

def cast_to_float(dic, dtype=np.float32):
    # cast to float 32 for space saving
    for key in dic.keys():
        dic[key] = dic[key].astype(dtype)
    return dic

def xyz2spherical(xyz):
    # convert cartesian coordinate to spherical coordinate
    # return in r, phi, and theta (elevation angle from z axis down)
    return_value = np.zeros(xyz.shape, dtype=xyz.dtype)
    xy = xyz[:,:,0]**2 + xyz[:,:,1]**2
    return_value[:,:,0] = np.sqrt(xy + xyz[:,:,2]**2) # r
    return_value[:,:,1] = np.arctan2(np.sqrt(xy), xyz[:,:,2]) # phi
    return_value[:,:,2] = np.arctan2(xyz[:,:,1], xyz[:,:,0]) #theta
    return return_value 

def spherical2xyz(rphitheta):
    return_value = np.zeros(rphitheta.shape, dtype=rphitheta.dtype)
    sinphi = np.sin(rphitheta[:,:,1])
    cosphi = np.cos(rphitheta[:,:,1])
    sintheta = np.sin(rphitheta[:,:,2])
    costheta = np.cos(rphitheta[:,:,2])
    return_value[:,:,0] = rphitheta[:,:,0]*sinphi*costheta # x
    return_value[:,:,1] = rphitheta[:,:,0]*sinphi*sintheta # y
    return_value[:,:,2] = rphitheta[:,:,0]*cosphi #z
    return return_value

# global variables
parent_idx = [0, 6, 7, \
              0, 1, 2, \
              0, 12, 13, 14,\
              13, 17, 18,\
              13, 25, 26]
child_idx = [6, 7, 8, \
             1, 2, 3, \
             12, 13, 14, 15,\
             17, 18, 19,\
             25, 26, 27]

def position_to_angle(skeletons):
    # transform 3d positions to joint angle representation 
    
    # first compute the bone vectors
    # a bone vector is the vector from on parent joint to one child joint 
    # hip->left hip->left knee->left foot,
    # hip->right hip-> right knee-> right foot
    # hip -> spine->thorax->nose->head
    # thorax -> left shoulder->left elbow->left wrist
    # thorax -> right shoulder-> right elbow->right wrist
    num_sample = skeletons.shape[0]
    skeletons = skeletons.reshape(num_sample, -1, 3)

    parent_joints = skeletons[:, parent_idx, :]
    child_joints = skeletons[:, child_idx, :]
    bone_vectors = child_joints - parent_joints
    # now compute the angles and bone lengths
    rphitheta = xyz2spherical(bone_vectors)    
    return rphitheta

def angle_to_position(rphitheta, skeletons):
    # transform joint angle representation to 3d positions 
    # starting from the root, create joint one by one according to predefined
    # hierarchical relation
    num_sample = skeletons.shape[0]
    skeletons = skeletons.reshape(num_sample, -1, 3)    
    for bone_idx in range(len(parent_idx)):
        offset = spherical2xyz(np.expand_dims(rphitheta[:, bone_idx, :], axis=1))
        offset = offset[:,0,:]
        skeletons[:, child_idx[bone_idx], :] = \
        skeletons[:, parent_idx[bone_idx], :] + offset
    return skeletons

def mutate_bone_length(population, opt, gen_idx, method='simple'):
    """
    Randomly mutate bone length in a population to increase variation in 
    subject size. 
    For example, H36M only contains adults yet you can modify bone
    length to represent children. Since the posture and subject size are 
    independent, you can synthetize data for dancing kids for free if you already
    have data for dancing adults. You only need little prior knowledge on human 
    bone length.
    """
    # the camera parameters in H36M correspond to the five subjects
    # Rename the synthetic population as these subjects so that the camera 
    # parameters can be used
    psuedo_subject_names = [1, 5, 6, 7, 8]
    dict_3d = {}
    for i in range(len(population)):
        if np.random.rand() > opt.MBLR:
            angles = position_to_angle(population[i].reshape(1, -1))  
            if method == 'simple':
                # The simplest way is to change to bone length to some value
                # according to prior knowledge about human bone size.
                # In our experiment, we collect these values manually from our 
                # interactive visualization tool as well as cross validation. 
                idx = np.random.randint(0, len(bl_templates))
                angles[0, :, 0] = bl_templates[idx]
                population[i] = (angle_to_position(angles, population[i].reshape(1,-1))).reshape(-1)
            elif method == 'addnoise':
                # add Gaussian noise to current bone length to obtain new bone length
                raise ValueError('Deprecated')
            else:
                raise NotImplementedError
    poses_list = np.array_split(population, len(psuedo_subject_names))
    for subject_idx in range(len(psuedo_subject_names)):
        dict_3d[(psuedo_subject_names[subject_idx], 'n/a', 'n/a')] =\
        poses_list[subject_idx]       
    save_path = get_save_path(opt, gen_idx)
    np.save(save_path, cast_to_float(dict_3d))
    logging.info('file saved at ' + save_path)
    return

def one_iteration(population, opt, model_file=None):
    """
    Run one iteration to produce the next generation.
    """
    # select the best individuals
    father_indices, mother_indices = choose_best(population, fraction=opt.F)
    # produce next generation by evolutionary operators
    offsprings = []
    for idx in tqdm(range(len(father_indices))):
        son, daughter = exploration(population[father_indices[idx]],
                                    population[mother_indices[idx]],
                                    opt)
        if son is not None:
            offsprings.append(son.reshape(1,-1))
        if daughter is not None:
            offsprings.append(daughter.reshape(1,-1))
    offsprings = np.concatenate(offsprings, axis=0)
    logging.info('{:d} out of {:d} poses survived.'.format(len(offsprings),
                 len(father_indices)*2))
    # select the synthetic data actively
    if opt.A:
        assert model_file is not None
        num_before = len(offsprings)
        offsprings, mean_error = active_select(model_file, offsprings, opt.AR)
        logging.info('{:d} out of {:d} poses are selected actively with mean'\
                     'error {:.2f}'.format(len(offsprings), num_before, mean_error))        
    if opt.Mer:
        # merge the offsprings with the parents
        population = np.vstack([population, offsprings])
    else:
        population = offsprings    
    return population

def get_save_path(opt, gen_idx):
    if opt.WS:
        save_path = os.path.join(opt.SD, opt.SS, opt.SN)
    else:
        save_path = os.path.join(opt.SD, 'S15678', opt.SN)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'generation_{:d}.npy'.format(gen_idx))
    return save_path

def split_and_save(final_poses, parameters, gen_idx):
    temp_subject_list = [1, 5, 6, 7, 8]
    train_set_3d = {}
    poses_list = np.array_split(final_poses, len(temp_subject_list))
    for subject_idx in range(len(temp_subject_list)):
        train_set_3d[(temp_subject_list[subject_idx], 'n/a', 'n/a')] =\
        poses_list[subject_idx]         
    save_path = get_save_path(parameters, gen_idx)
    np.save(save_path, cast_to_float(train_set_3d))     
    print('file saved at {:s}!'.format(save_path))
    return

def save_results(poses, opt, gen_idx):
    # get save path
    if opt.MBL:
        mutate_bone_length(poses, opt, gen_idx)
    else:
        split_and_save(poses, opt, gen_idx)
    return

def evolution(initial_population, opt, model_file=None):
    """
    Dataset evolution.
    """
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s]: %(message)s"
                        )
    population = initial_population
    save_results(initial_population, opt, 0)
    initial_num = len(initial_population)
    for gen_idx in range(1, opt.G+1):
        population = one_iteration(population, opt, model_file=model_file)
    save_results(population, opt, gen_idx)
    # if not enough
    if opt.E and len(population) < initial_num * opt.T:
        logging.info('Running extra generations to synthesize enough data...')
        while len(population) < initial_num * opt.T:
            gen_idx += 1
            logging.info('Generation {:d}...'.format(gen_idx))
            population = one_iteration(population, opt, model_file=model_file)
            if opt.I:
                save_results(population.copy(), opt, gen_idx)
                logging.info('Generation {:d} saved.'.format(gen_idx))
    save_results(population, opt, gen_idx)
    logging.info('Final population saved.')
    return population
