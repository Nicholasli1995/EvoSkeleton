"""
Utility functions for dealing with Human3.6M dataset.
Some functions are adapted from https://github.com/una-dinosauria/3d-pose-baseline 
"""

import os
import numpy as np
import copy
import logging
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D

import libs.dataset.h36m.cameras as cameras
import libs.dataset.h36m.pth_dataset as dataset
# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS  = [9, 11]

# Use camera coordinate system 
camera_frame = True

# Joint names in H3.6M -- data has 32 joints, but only 17 that move; 
# these are the indices.
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

parent_indices = np.array([0, 1, 2, 0, 6, 7, 0, 12, 13, 14, 13, 17, 18, 13, 25, 26])
children_indices = np.array([1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27])

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

# The .h5 suffix in pose sequence name is just inherited from the original 
# naming convention. The '.sh' suffix means stacked hourglass key-point detector 
# used in previous works. Here we just use '.sh' to represent key-points obtained
# from any heat-map regression model. We used high-resolution net instead of 
# stacked-hourglass model.  

def load_ckpt(opt):
    cascade = torch.load(os.path.join(opt.ckpt_dir, 'model.th'))
    stats = np.load(os.path.join(opt.ckpt_dir, 'stats.npy'), allow_pickle=True).item()
    if opt.cuda:
        cascade.cuda()
    return cascade, stats

def list_remove(list_a, list_b):
    """
    Fine all elements of a list A that does not exist in list B.
    
    Args
      list_a: list A
      list_b: list B
    Returns
      list_c: result  
    """
    list_c = []
    for item in list_a:
        if item not in list_b:
            list_c.append(item)
    return list_c

def add_virtual_cams(cams, visualize=False):
    """
    Deprecated. Add virtual cameras.
    """
    # add more cameras to the scene
    #R, T, f, c, k, p, name = cams[ (1,1) ]
    # plot the position of human subjects
    old_cam_num = 4
    def add_coordinate_system(ax, origin, system, length=300, new=False):
        # draw a coordinate system at a specified origin
        origin = origin.reshape(3, 1)
        start_points = np.repeat(origin, 3, axis=1)
        # system: [v1, v2, v3] 
        end_points = start_points + system*length
        color = ['g', 'y', 'k'] # color for v1, v2 and v3
        if new:
            color = ['b', 'r', 'g']
        def get_args(start_points, end_points):
            x = [start_points[0], end_points[0]]
            y = [start_points[1], end_points[1]]
            z = [start_points[2], end_points[2]]
            return x, y, z
        for i in range(3):
            x, y, z = get_args(start_points[:,i], end_points[:,i])
            ax.plot(x, y, z,  lw=2, c=color[i])
        return
    
    def get_new_camera(system, center, rotation = [0,0,90.]):
        from scipy.spatial.transform import Rotation as Rotation
        center = center.reshape(3, 1)
        start_points = np.repeat(center, 3, axis=1)
        end_points = start_points + system
        r = Rotation.from_euler('xyz', rotation, degrees=True)
        start_points_new = r.as_dcm() @ start_points  
        end_points_new = r.as_dcm() @ end_points
        new_system = [(end_points_new[:,i] - start_points_new[:,i]).reshape(3,1) for i in range(3)]
        new_system = np.hstack(new_system)
        return new_system, start_points_new[:,0]    

    
    # the new cameras are added by rotating one existing camera
    # TODO: more rotations
    new_cams = cams.copy()
    for key in cams.keys():
        subject, camera_idx = key
        if camera_idx != 1: # only rotate the first camera
            continue
        R, T, f, c, k, p, name = cams[key]
        angles = [80., 130., 270., 320.]
        for angle_idx in range(len(angles)):
            angle = angles[angle_idx]
            new_R, new_T = get_new_camera(R.T, T, [0., 0., angle])
            new_cams[(subject, old_cam_num + angle_idx + 1)]\
            = (new_R.T, new_T.reshape(3,1), f, c, k, p, name+'new'+str(angle_idx+1))
    # visualize cameras used
    if visualize:
        train_set_3d = np.load('../data/human3.6M/h36m/numpy/threeDPose_train.npy').item()
        test_set_3d = np.load('../data/human3.6M/h36m/numpy/threeDPose_test.npy').item()
        hips_train = np.vstack(list(train_set_3d.values()))
        hips_test = np.vstack(list(test_set_3d.values()))        
        ax = plt.subplot(111, projection='3d')
        chosen = np.random.choice(len(hips_train), 1000, replace=False)
        chosen_hips = hips_train[chosen, :3]
        ax.plot(chosen_hips[:,0], chosen_hips[:,1], chosen_hips[:,2], 'bo')
        chosen = np.random.choice(len(hips_test), 1000, replace=False)
        chosen_hips = hips_test[chosen, :3]
        ax.plot(chosen_hips[:,0], chosen_hips[:,1], chosen_hips[:,2], 'ro')        
        ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
        plt.title('Blue dots: Hip positions in the h36m training set. \
                  Red dots: testing set. \
                  Old camera coordinates: x-green, y-yellow, z-black \
                  New camera coordinates: x-blue, y-red, z-green')
        plt.pause(0.1)
        for key in new_cams.keys():
            R, T, f, c, k, p, name = new_cams[key]
            # R gives camera basis vectors row-by-row, T gives camera center
            if 'new' in name:
                new = True
            else:
                new = False
            add_coordinate_system(ax, T, R.T, new=new)
        RADIUS = 3000 # space around the subject
        xroot, yroot, zroot = 0., 0., 500.
        ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
        ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
        ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
        ax.set_aspect("equal")
    return new_cams

def down_sample_training_data(train_dict, opt):
    """
    Down-sample the training data.

    Args
      train_dict: python dictionary contraining the training data
      opt: experiment options
    Returns
      train_dict/sampled_dict: a dictionary containing a subset of training data
    """
    if opt.ws_name in ['S1', 'S15', 'S156']:
        sub_list = [int(opt.ws_name[i]) for i in range(1, len(opt.ws_name))]
        keys_to_delete = []
        for key in train_dict.keys():
            if key[0] not in sub_list:
                keys_to_delete.append(key)
        for key in keys_to_delete:    
            del train_dict[key]
        return train_dict
    elif opt.ws_name in ['0.001S1','0.01S1', '0.05S1', '0.1S1', '0.5S1']:
        ratio = float(opt.ws_name.split('S')[0])
        # randomly sample a portion of 3D data
        sampled_dict = {}
        for key in train_dict.keys():
            if key[0] != 1:
                continue
            total = len(train_dict[key])
            sampled_num = int(ratio*total)
            chosen_indices = np.random.choice(total, sampled_num, replace=False)
            sampled_dict[key] = train_dict[key][chosen_indices].copy()
        return sampled_dict
    else:
        raise ValueError('Unknown experiment setting.')
    
def get_train_dict_3d(opt):
    """
    Get the training 3d skeletons as a Python dictionary.

    Args
      opt: experiment options
    Returns
      train_dict_3d: a dictionary containing training 3d poses
    """
    if not opt.train:
        return None
    dict_path = os.path.join(opt.data_dir, 'threeDPose_train.npy')
    #=========================================================================#
    # For real 2D detections, the down-sampling and data augmentation 
    # are done later in get_train_dict_2d
    if opt.twoD_source != 'synthetic':
        train_dict_3d = np.load(dict_path, allow_pickle=True).item()
        return train_dict_3d
    #=========================================================================#
    # For synthetic 2D detections (For Protocol P1*), the down-sampling is 
    # performed here and the data augmentation is assumed to be already done
    if opt.evolved_path is not None:
        # the data is pre-augmented
        train_dict_3d = np.load(opt.evolved_path, allow_pickle=True).item()
    elif opt.ws:
        # raw training data from Human 3.6M (S15678) 
        # Down-sample the raw data to simulate an environment with scarce 
        # training data, which is used in weakly-supervised experiments        
        train_dict_3d = np.load(dict_path, allow_pickle=True).item()
        train_dict_3d = down_sample_training_data(train_dict_3d, opt)
    else:
        # raw training data from Human 3.6M (S15678) 
        train_dict_3d = np.load(dict_path, allow_pickle=True).item()
    return train_dict_3d

def get_test_dict_3d(opt):
    """
    Get the testing 3d skeletons as a Python dictionary.

    Args
      opt: experiment options
    Returns
      test_dict_3d: a dictionary containing testing 3d poses
    """       
    if opt.test_source == 'h36m':     
        # for h36m
        dict_path = os.path.join(opt.data_dir, 'threeDPose_test.npy')
        test_dict_3d  = np.load(dict_path, allow_pickle=True).item()   
    else:
        raise NotImplementedError    
    return test_dict_3d

def get_dict_2d(train_dict_3d, test_dict_3d, rcams, ncams, opt):
    """
    Prepare 2D training and testing data as Python dictionaries.

    Args
      train_dict_3d: dictionary containing training 3d poses
      test_dict_3d: dictionary containing testing 3d poses
      rcams: camera parameters
      ncams: number of camera to use
      opt: experiment options
    Returns
      train_dict_2d: a dictionary containing training 2d poses
      test_dict_2d: a dictionary containing testing 2d poses
      train_dict_3d: the dictionary containing training 3d poses, which may be
      updated
    """     
    if opt.twoD_source == 'synthetic':
        # project the 3D key-points to 2D ones
        # This type of key-points is used to validate the performance of 
        # 2D-to-3D networks and the noise of 2D key-point detector is ignored.
        # In fact, these 2D key-points are used as ground-truth to train the 
        # first stage of TAG-net.
        if opt.virtual_cams:
            ncams *= 2       
        if opt.train:
            train_dict_2d = project_to_cameras(train_dict_3d, rcams, ncams=ncams)
        else:
            train_dict_2d = None
        test_dict_2d  = project_to_cameras(test_dict_3d, rcams, ncams=ncams)
    elif opt.twoD_source == 'HRN':
        # The 2D key-point detections obtained by the heatmap regression model.
        # The model uses high-resolution net as backbone and pixel-shuffle super-resolution
        # to regress high-resolution heatmaps.
        if opt.train:
            train_dict_2d = np.load(os.path.join(opt.data_dir, 'twoDPose_HRN_train.npy'), allow_pickle=True).item()           
        else:
            train_dict_2d = None
        test_dict_2d = np.load(os.path.join(opt.data_dir, 'twoDPose_HRN_test.npy'), allow_pickle=True).item()
        
        def delete(dic, actions):
            keys_to_delete = []
            for key in dic.keys():
                sub, act, name = key
                if act not in actions:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del dic[key]
            return dic

        def replace(dic, temp):
            for key in dic.keys():
                sub, act, name = key
                temp_key = (sub, act, name[:-3])
                synthetic = temp[temp_key]
                assert len(dic[key]) == len(synthetic)
                indices = np.random.choice(len(synthetic), int(0.5*len(synthetic)), replace=False)
                dic[key][indices] = synthetic[indices].copy()
            return dic

#        # weakly-supervised experiment
        def remove_keys(dic, name_list):
            keys_to_delete = []
            for key in dic.keys():
                if key[0] not in name_list:
                    keys_to_delete.append(key)
            for key in keys_to_delete:    
                del dic[key]
            return dic
        
        # down-sample the data for weakly-supervised experiment
        if opt.ws and opt.ws_name in ['S1', 'S15', 'S156']:
            sub_list = [int(opt.ws_name[i]) for i in range(1, len(opt.ws_name))]
            remove_keys(train_dict_3d, sub_list)
            if train_dict_2d is not None:
                remove_keys(train_dict_2d, sub_list)
        # data augmentation with evolved data
        if opt.evolved_path is not None:
            evolved_dict_3d = np.load(opt.evolved_path, allow_pickle=True).item()
            evolved_dict_2d = project_to_cameras(evolved_dict_3d, rcams, ncams=ncams)
            # combine the synthetic 2D-3D pair with the real 2D-3D pair
            train_dict_3d = {**train_dict_3d, **evolved_dict_3d}
            train_dict_2d = {**train_dict_2d, **evolved_dict_2d} 
    return train_dict_2d, test_dict_2d, train_dict_3d        

def prepare_data_dict(rcams, 
                      opt,
                      ncams=4,
                      predict_14=False, 
                      use_nose=True
                      ):
    """
    Prepare 2D and 3D data as Python dictionaries.
    
    Args
      rcams: camera parameters
      opt: experiment options
      ncams: number of camera to use
      predict_14: whether to predict 14 joints or not
      use_nose: whether to use nose joint or not
    Returns
      data_dic: a dictionary containing training and testing data
      data_stats: statistics computed from training data 
    """
    assert opt.twoD_source in ['synthetic', 'HRN'], 'Unknown 2D key-point type.'
    data_dic = {}
    # get 3D skeleton data
    train_dict_3d = get_train_dict_3d(opt)
    test_dict_3d = get_test_dict_3d(opt)
    # get 2D key-point data
    train_dict_2d, test_dict_2d, train_dict_3d = get_dict_2d(train_dict_3d, 
                                                             test_dict_3d, 
                                                             rcams, 
                                                             ncams,
                                                             opt
                                                             )
    # compute normalization statistics and normalize the 2D data
    if opt.train:
        complete_train_2d = copy.deepcopy(np.vstack(list(train_dict_2d.values())))
        data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = \
        normalization_stats(complete_train_2d, 
                            dim=2, 
                            norm_twoD=opt.norm_twoD, 
                            use_nose=use_nose
                            )
        
        data_dic['train_set_2d'] = normalize_data(train_dict_2d,
                                                  data_mean_2d,
                                                  data_std_2d, 
                                                  dim_to_use_2d, 
                                                  norm_single=opt.norm_single
                                                  )
    else:
        _, data_stats = load_ckpt(opt)
        data_mean_2d, data_std_2d = data_stats['mean_2d'], data_stats['std_2d']
        dim_to_use_2d = data_stats['dim_use_2d']
    data_dic['test_set_2d']  = normalize_data(test_dict_2d,
                                              data_mean_2d,
                                              data_std_2d, 
                                              dim_to_use_2d, 
                                              norm_single=opt.norm_single
                                              )
    # The 3D joint position is represented in the world coordinate,
    # which is converted to camera coordinate system as the regression target
    if opt.train:
        train_dict_3d = transform_world_to_camera(train_dict_3d, rcams, ncams=ncams)
        # apply 3d post-processing (centering around root)
        train_dict_3d, train_root_positions = postprocess_3d(train_dict_3d)
    test_dict_3d  = transform_world_to_camera(test_dict_3d, rcams, ncams=ncams)
    test_dict_3d,  test_root_positions  = postprocess_3d(test_dict_3d)
    if opt.train:
        # compute normalization statistics and normalize the 3D data
        complete_train_3d = copy.deepcopy(np.vstack(list(train_dict_3d.values())))
        data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d =\
        normalization_stats(complete_train_3d, dim=3, predict_14=predict_14)
        data_dic['train_set_3d'] = normalize_data(train_dict_3d,
                                                  data_mean_3d,
                                                  data_std_3d, 
                                                  dim_to_use_3d
                                                  )
        # some joints are not used during training
        dim_use_2d = list_remove([i for i in range(len(data_mean_2d))], 
                                  list(dim_to_ignore_2d))
        dim_use_3d = list_remove([i for i in range(len(data_mean_3d))], 
                                  list(dim_to_ignore_3d))
        # assemble a dictionary for data statistics
        data_stats = {'mean_2d':data_mean_2d, 
                      'std_2d':data_std_2d,
                      'mean_3d':data_mean_3d, 
                      'std_3d':data_std_3d,
                      'dim_ignore_2d':dim_to_ignore_2d, 
                      'dim_ignore_3d':dim_to_ignore_3d,
                      'dim_use_2d':dim_use_2d,
                      'dim_use_3d':dim_use_3d
                      }         
    else:
        data_mean_3d, data_std_3d = data_stats['mean_3d'], data_stats['std_3d']
        dim_to_use_3d = data_stats['dim_use_3d']            
    data_dic['test_set_3d']  = normalize_data(test_dict_3d,  
                                              data_mean_3d,
                                              data_std_3d, 
                                              dim_to_use_3d
                                              )    
   
    return data_dic, data_stats

def select_action(dic_2d, dic_3d, action, twoD_source):
    """
    Construct sub-dictionaries by specifying which action to use
    
    Args
        dic_2d: dictionary containing 2d poses
        dic_3d: dictionary containing 3d poses
        action: the action to use
        twoD_source: how the key-points are generated (synthetic or real)
    Returns
        dic_2d_action: sub-dictionary containing 2d poses for the specified action
        dic_3d_action: sub-dictionary containing 3d poses for the specified action 
    """    
    dic_2d_action = {}
    dic_3d_action = {}
    for key in dic_2d.keys():
        if key[1] == action:
            dic_2d_action[key] = dic_2d[key].copy()
            if twoD_source == 'synthetic':
                key3d = key
            else:
                key3d = (key[0], key[1], key[2][:-3])
            dic_3d_action[key3d] = dic_3d[key3d].copy()    
    return dic_2d_action, dic_3d_action

def split_action(dic_2d, dic_3d, actions, camera_frame, opt, input_size, output_size):
    """
    Generate a list of datasets for each action.
    
    Args
        dic_2d: dictionary containing 2d poses
        dic_3d: dictionary containing 3d poses
        actions: list of defined actions
        camera_frame: use camera coordinate system
        opt: experiment options
        input_size: input vector length
        output_size: output vector length
    Returns
        action_dataset_list: a list of datasets where each element correspond
        to one action   
    """
    action_dataset_list = []
    for act_id in range(len(actions)):
        action = actions[act_id]
        dic_2d_action, dic_3d_action = select_action(dic_2d, dic_3d, action, opt.twoD_source)
        eval_input, eval_output = get_all_data(dic_2d_action, 
                                               dic_3d_action,
                                               camera_frame, 
                                               norm_twoD=opt.norm_twoD,
                                               input_size=input_size,
                                               output_size=output_size)
        action_dataset = dataset.PoseDataset(eval_input, 
                                             eval_output, 
                                             'eval', 
                                             action_name=action,
                                             refine_3d=opt.refine_3d)
        action_dataset_list.append(action_dataset)
    return action_dataset_list

def normalization_stats(complete_data, 
                        dim, 
                        predict_14=False, 
                        norm_twoD=False, 
                        use_nose=False
                        ):
    """
    Computes normalization statistics: mean and stdev, dimensions used and ignored
    
    Args
        complete_data: nxd np array with poses
        dim. integer={2,3} dimensionality of the data
        predict_14. boolean. Whether to use only 14 joints
        use_nose: whether to use nose or not
    Returns
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dimensions_to_ignore: list of dimensions not used in the model
        dimensions_to_use: list of dimensions used in the model
    """
    if not dim in [2,3]:
        raise(ValueError, 'dim must be 2 or 3')
    data_mean = np.mean(complete_data, axis=0)
    data_std  =  np.std(complete_data, axis=0)
    # Encodes which 17 (or 14) 2d-3d pairs we are predicting
    dimensions_to_ignore = []
    if dim == 2:
        if not use_nose:
            dimensions_to_use = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        else:
            dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]        
        if norm_twoD:
            dimensions_to_use = np.delete(dimensions_to_use, 0)
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use*2, 
                                               dimensions_to_use*2+1)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES)*2), 
                                         dimensions_to_use)
    else: 
        dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        # hip is deleted
        # spine and neck are also deleted if predict_14 
        dimensions_to_use = np.delete(dimensions_to_use, [0,7,9] if predict_14 else 0)
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use*3,
                                               dimensions_to_use*3+1,
                                               dimensions_to_use*3+2)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES)*3), 
                                         dimensions_to_use)
    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use

def transform_world_to_camera(poses_set, cams, ncams=4):
    """
    Transform 3d poses from world coordinate to camera coordinate system
    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted(poses_set.keys()):
      subj, action, seqname = t3dk
      t3d_world = poses_set[t3dk]
      for c in range(ncams):
        R, T, f, c, k, p, name = cams[(subj, c+1)]
        camera_coord = cameras.world_to_camera_frame(np.reshape(t3d_world, [-1, 3]), R, T)
        camera_coord = np.reshape(camera_coord, [-1, len(H36M_NAMES)*3])
        sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
        t3d_camera[(subj, action, sname)] = camera_coord
    return t3d_camera

def normalize_data(data, data_mean, data_std, dim_to_use, norm_single=False):
    """
    Normalizes a dictionary of poses
    
    Args
        data: dictionary where values are
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dim_to_use: list of dimensions to keep in the data
        norm_single: whether to perform normalization independently for each 
        sample
    Returns
        data_out: dictionary with same keys as data, but values have been normalized
    """
    data_out = {}
    for key in data.keys():
        data[ key ] = data[ key ][ :, dim_to_use ]
        if norm_single:
            # does not use statistics over the whole dataset
            temp = data[key]
            temp = temp.reshape(len(temp), -1, 2)
            mean_x = np.mean(temp[:,:,0], axis=1).reshape(len(temp), 1)
            std_x = np.std(temp[:,:,0], axis=1)
            mean_y = np.mean(temp[:,:,1], axis=1).reshape(len(temp), 1)
            std_y = np.std(temp[:,:,1], axis=1)
            denominator = (0.5*(std_x + std_y)).reshape(len(std_x), 1)
            temp[:,:,0] = (temp[:,:,0] - mean_x)/denominator
            temp[:,:,1] = (temp[:,:,1] - mean_y)/denominator
            data_out[key] = temp.reshape(len(temp), -1)
        else:
            mu = data_mean[dim_to_use]
            stddev = data_std[dim_to_use]
            data_out[ key ] = np.divide( (data[key] - mu), stddev )    
    return data_out

def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):
    """
    Un-normalizes a matrix whose mean has been substracted and that has been 
    divided by standard deviation. Some dimensions might also be missing.
    
    Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
    Returns
    orig_data: the unnormalized data
    """
    T = normalized_data.shape[0] # batch size
    D = data_mean.shape[0] # dimensionality
    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = np.array([dim for dim in range(D)
                                    if dim not in dimensions_to_ignore])
    orig_data[:, dimensions_to_use] = normalized_data
    # multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data

def define_actions(action):
    """
    Given an action string, returns a list of corresponding actions.
    
    Args
        action: String. either "all" or one of the h36m actions
    Returns
        actions: List of strings. Actions to use.
    Raises
        ValueError: if the action is not a valid action in Human 3.6M
    """
    actions = ["Directions",
               "Discussion",
               "Eating",
               "Greeting",
               "Phoning",
               "Photo",
               "Posing",
               "Purchases",
               "Sitting",
               "SittingDown",
               "Smoking",
               "Waiting",
               "WalkDog",
               "Walking",
               "WalkTogether"
               ]
    
    if action == "All" or action == "all":
        return actions
    
    if not action in actions:
        raise( ValueError, "Unrecognized action: %s" % action )
    
    return [action]

def project_to_cameras(poses_set, cams, ncams=4):
    """
    Project 3d poses using camera parameters
    
    Args
        poses_set: dictionary containing 3d poses
        cams: dictionary containing camera parameters
        ncams: number of cameras per subject
    Returns
        t2d: dictionary with 2d poses
    """
    t2d = {}
    for t3dk in sorted(poses_set.keys()):
        subj, a, seqname = t3dk
        t3d = poses_set[t3dk]
        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, cam+1)]
            pts2d, _, _, _, _ = cameras.project_point_radial(np.reshape(t3d, [-1, 3]), R, T, f, c, k, p)        
            pts2d = np.reshape(pts2d, [-1, len(H36M_NAMES)*2])
            sname = seqname[:-3] + "." + name + ".h5" # e.g.: Waiting 1.58860488.h5
            t2d[ (subj, a, sname) ] = pts2d
    return t2d

def postprocess_3d(poses_set):
    """
    Center 3d points around root
    
    Args
        poses_set: dictionary with 3d data
    Returns
        poses_set: dictionary with 3d data centred around root (center hip) joint
        root_positions: dictionary with the original 3d position of each pose
    """
    root_positions = {}
    for k in poses_set.keys():
        # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:,:3])
        # Remove the root from the 3d position
        poses = poses_set[k]
        poses = poses - np.tile( poses[:,:3], [1, len(H36M_NAMES)] )
        poses_set[k] = poses
    return poses_set, root_positions

def postprocess_2d(poses_set):
    """
    Center 2d points around root
    
    Args
        poses_set: dictionary with 2d data
    Returns
        poses_set: dictionary with 2d data centred around root (center hip) joint
        root_positions: dictionary with the original 2d position of each pose
    """
    root_positions = {}
    for k in poses_set.keys():
    # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:,:2])
        # Remove the root from the 3d position
        poses = poses_set[k]
        poses = poses - np.tile( poses[:,:2], [1, len(H36M_NAMES)] )
        poses_set[k] = poses
    return poses_set, root_positions

def get_all_data(data_x, 
                 data_y, 
                 camera_frame, 
                 norm_twoD=False, 
                 input_size=32,
                 output_size=48
                 ):
    """
    Obtain numpy arrays for network inputs/outputs
    
    Args
      data_x: dictionary with 2d inputs
      data_y: dictionary with 3d expected outputs
      camera_frame: whether the 3d data is in camera coordinates
      input_size: input vector length for each sample
      output_size: output vector length for each sample
    Returns
      encoder_inputs: numpy array for the input data 
      decoder_outputs: numpy array for the output data
    """
    if norm_twoD:
        input_size -= 2
    # Figure out how many frames we have
    n = 0
    for key2d in data_x.keys():
      n2d, _ = data_x[ key2d ].shape
      n = n + n2d

    encoder_inputs  = np.zeros((n, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((n, output_size), dtype=np.float32)

    # Put all the data into big arrays
    idx = 0
    for key2d in data_x.keys():
      (subj, b, fname) = key2d
      # keys should be the same if 3d is in camera coordinates
      key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
      # '-sh' suffix means detected key-points are used 
      key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d

      n2d, _ = data_x[ key2d ].shape
      encoder_inputs[idx:idx+n2d, :]  = data_x[ key2d ]
      decoder_outputs[idx:idx+n2d, :] = data_y[ key3d ]
      idx = idx + n2d

    return encoder_inputs, decoder_outputs

def prepare_dataset(opt):
    """
    Prepare PyTorch dataset objects used for training 2D-to-3D deep network
    
    Args
        opt: experiment options
    Returns
        train_dataset: training dataset as PyTorch dataset object
        eval_dataset: evaluation dataset as PyTorch dataset object 
        data_stats: dataset statistics computed from the training dataset
        action_eval_list: a list of evaluation dataset objects where each 
        corresponds to one action
    """
    # get relevant paths
    data_dir =  opt.data_dir
    cameras_path = os.path.join(data_dir, 'cameras.npy')
    # By default, all actions are used
    actions = define_actions(opt.actions)
    # load camera parameters to project 3D skeleton
    rcams = np.load(cameras_path, allow_pickle=True).item()    
    # produce more camera views by adding virtual cameras if needed
    if opt.virtual_cams:
        rcams = add_virtual_cams(rcams)
    # first prepare Python dictionary containing 2D and 3D data
    data_dic, data_stats = prepare_data_dict(rcams, 
                                             opt,
                                             predict_14=False
                                             )
    input_size = len(data_stats['dim_use_2d'])
    output_size = len(data_stats['dim_use_3d'])
    if opt.train:
        # convert Python dictionary to numpy array
        train_input, train_output = get_all_data(data_dic['train_set_2d'],
                                                 data_dic['train_set_3d'], 
                                                 camera_frame,
                                                 norm_twoD=opt.norm_twoD,
                                                 input_size=input_size,
                                                 output_size=output_size
                                                 )
        # The Numpy arrays are finally used to initialize the dataset objects
        train_dataset = dataset.PoseDataset(train_input, 
                                            train_output, 
                                            'train',
                                            refine_3d = opt.refine_3d
                                            )
    else:
        train_dataset = None
            
    eval_input, eval_output = get_all_data(data_dic['test_set_2d'],
                                           data_dic['test_set_3d'], 
                                           camera_frame,
                                           norm_twoD=opt.norm_twoD,
                                           input_size=input_size,
                                           output_size=output_size
                                           )

    eval_dataset = dataset.PoseDataset(eval_input, 
                                       eval_output, 
                                       'eval',
                                       refine_3d = opt.refine_3d
                                       )
    # Create a list of dataset objects for action-wise evaluation
    action_eval_list = split_action(data_dic['test_set_2d'],
                                    data_dic['test_set_3d'],
                                    actions, 
                                    camera_frame,
                                    opt,
                                    input_size=input_size,
                                    output_size=output_size
                                    )
    
    return train_dataset, eval_dataset, data_stats, action_eval_list