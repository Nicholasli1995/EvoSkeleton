"""
Evolution of 3D human skeleton.
author: Nicholas Li
contact: nicholas.li@connect.ust.hk
"""
import sys
sys.path.append("../")

from libs.evolution.genetic import evolution
from libs.evolution.parameter import parse_arg

import os
import logging
from scipy.spatial.transform import Rotation as R
import numpy as np

def cast_to_float(dic, dtype=np.float32):
    # cast to float 32 for space saving
    for key in dic.keys():
        dic[key] = dic[key].astype(dtype)
    return dic

def random_rotation(pose, sigma=360):
    # apply random rotation to equivalently augment viewpoints
    pose = pose.reshape(32, 3)
    hip = pose[0].copy().reshape(1, 3)
    x = np.random.normal(scale=sigma)
    y = np.random.normal(scale=sigma)
    z = np.random.normal(scale=sigma)
    r = R.from_euler('xyz', [x, y, z], degrees=True)
    rotated = r.as_dcm() @ (pose-hip).T
    return (rotated.T + hip).reshape(-1)

def initialize_population(data_dic, opt):
    """
    Initialize a population for later evolution.
    """
    # down-sample the raw data if used for weakly-supervised experiments
    if opt.WS and opt.SS.startswith("0.") and opt.SS.endswith("S1"):
        # a fraction of S1 data for H36M
        ratio = float(opt.SS.split('S')[0])
        # randomly sample a portion of 3D data
        sampled_dic = {}
        # sample each video
        for key in data_dic.keys():
            if key[0] != 1:
                continue
            total = len(data_dic[key])
            sampled_num = int(ratio*total)
            chosen_indices = np.random.choice(total, sampled_num, replace=False)
            sampled_dic[key] = data_dic[key][chosen_indices].copy()
        initial_population = np.concatenate(list(sampled_dic.values()), axis=0)   
    elif opt.WS and opt.SS.startswith("S"):
        # a collection of data from a few subjects
        # delete unused subjects
        sub_list = [int(opt.SS[i]) for i in range(1, len(opt.SS))]
        keys_to_delete = []
        for key in data_dic.keys():
            if key[0] not in sub_list:
                keys_to_delete.append(key)
        for key in keys_to_delete:    
            del data_dic[key]        
        initial_population = np.concatenate(list(data_dic.values()), axis=0)               
    else:
        # do not perform down-sampling
        initial_population = np.concatenate(list(data_dic.values()), axis=0)    
    return initial_population

def initialize_model_file(opt):
    if opt.A:
        import torch
        model = torch.load(os.path.join(opt.ckpt_dir, "model.th"))
        stats = np.load(os.path.join(opt.ckpt_dir, "stats.npy")).item()
        cameras = np.load("../data/human3.6M/cameras.npy").item()
        model_file = {"model":model, "stats":stats, "cams":list(cameras.values())}
    else:
        model_file = None  
    return model_file


def split_and_save(evolved_population):
    """
    Split and save the evolved dataset into training and validation set.
    """
    training_indices = np.random.choice(len(evolved_population), int(0.95*len(evolved_population)), replace=False)
    testing_indices = np.delete(np.arange(len(evolved_population)), training_indices)
    training_poses = evolved_population[training_indices]
    testing_poses = evolved_population[testing_indices]

    temp_subject_list = [1, 5, 6, 7, 8]
    train_set_3d = {}
    poses_list = np.array_split(training_poses, len(temp_subject_list))
    for subject_idx in range(len(temp_subject_list)):
        train_set_3d[(temp_subject_list[subject_idx], 'n/a', 'n/a')] =\
        poses_list[subject_idx] 
    # testing
    testing_poses = evolved_population[testing_indices]
    temp_subject_list = [9,11]
    test_set_3d = {}
    poses_list = np.array_split(testing_poses, len(temp_subject_list))
    for subject_idx in range(len(temp_subject_list)):
        test_set_3d[(temp_subject_list[subject_idx], 'n/a', 'n/a')] =\
        poses_list[subject_idx] 
    np.save('../data/human3.6M/h36m/numpy/threeDPose_train_split.npy', train_set_3d)
    np.save('../data/human3.6M/h36m/numpy/threeDPose_test.npy', test_set_3d)    
    return

def visualize(initial_population, evolved_population):
    """
    Visualize the augmented dataset
    """
    import matplotlib.pyplot as plt
    from genetic import show3Dpose
    def get_zmin(pose):
        return pose.reshape(32,3)[:,2].min()
    # initial population
    chosen_indices = np.random.choice(len(initial_population), 9, replace=False)
    plt.figure()
    for idx in range(9):
        ax = plt.subplot(3, 3, idx+1, projection='3d')
        pose = initial_population[chosen_indices[idx]]
        show3Dpose(pose, ax) 
        plt.title("{:d}:{:.2f}".format(chosen_indices[idx], get_zmin(pose)))
    plt.tight_layout() 
    # after evolution
    chosen_indices = np.random.choice(len(evolved_population) - len(initial_population), 9, replace=False)
    plt.figure()
    for idx in range(9):
        ax = plt.subplot(3, 3, idx+1, projection='3d')
        pose = evolved_population[chosen_indices[idx] + len(initial_population)]
        show3Dpose(pose, ax) 
        plt.title("{:d}:{:.2f}".format(chosen_indices[idx] + len(initial_population), get_zmin(pose)))
    plt.tight_layout()     
    return

def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s]: %(message)s"
                        ) 
    # parse command line input
    opt = parse_arg()    
    if opt.generate:
        # get the training set of human 3.6M 
        data_dic = np.load(opt.data_path, allow_pickle=True).item()
        initial_population = initialize_population(data_dic, opt)
        # load a pre-trained model for active searching (optional)
        model_file = initialize_model_file(opt)      
        evolved_population = evolution(initial_population,
                                       opt,
                                       model_file=model_file
                                       )        
        if opt.split:
            split_and_save(evolved_population)
    
    if opt.visualize:
        visualize(initial_population, evolved_population)
    
if __name__ == "__main__":
    main()