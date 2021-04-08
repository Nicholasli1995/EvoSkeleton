import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='2Dto3Dnet.py')
    ## paths
    parser.add_argument('-save_root', type=str, default='../model/')
    ##-----------------------------------------------------------------------##
    ## model settings
    parser.add_argument('-save_name', type=str, default=None)
    # feed the current estimated 3D poses to the next stage
    parser.add_argument('-refine_3d', type=bool, default=False)
    parser.add_argument('-norm_twoD', type=bool, default=False)
    parser.add_argument('-num_blocks', type=int, default=2)
    # how many stages used for boosted regression
    parser.add_argument('-num_stages', type=int, default=2)   
    # the length of 3D pose representation used in the network
    parser.add_argument('-linear_size', type=int, default=1024)  
    # extra name for logging
    parser.add_argument('-extra_str', type=str, default='')
    # dropout
    parser.add_argument('-dropout', type=float, default=0.5)
    # leaky ReLu
    parser.add_argument('-leaky', type=bool, default=False)
    ##-----------------------------------------------------------------------##    
    ## training settings
    parser.add_argument('-batch_size', type=int, default=8192)
    # random seed for reproduction of experiments
    parser.add_argument('-seed', type=int, default=2019)
    # number of threads to use when loading data
    parser.add_argument('-num_threads', type=int, default=4)
    # update leaf node distribution every certain number of network training
    parser.add_argument('-gpuid', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=200)
    # report_every: 10
    parser.add_argument('-report_every', type=int, default=100)
    # whether to perform evaluation on evaluation set during training
    parser.add_argument('-eval', type=bool, default=False)
    # whether to evaluate for each action during the training
    parser.add_argument('-eval_action_wise', type=bool, default=True)   
    # what protocol to use for evaluation
    parser.add_argument('-protocols', type=list, default=['P1', 'P2'])
    # whether to record and report loss history at the end of training  
    parser.add_argument('-eval_every', type=int, default=350)
    # path to the human3.6M dataset
    parser.add_argument('-data_dir', type=str, default='../data/human3.6M/')
    # actions to use for training
    parser.add_argument('-actions', type=str, default='All')
    # whether to do data augmentation for the training data
    parser.add_argument('-augmentation', type=bool, default=False)    
    # using virtual cameras
    parser.add_argument('-virtual_cams', type=bool, default=False)    
    # interpolate between 3D joints
    parser.add_argument('-interpolate', type=bool, default=False)    
    # what input to use, synthetic or detected
    parser.add_argument('-twoD_source', type=str, default='synthetic')
    # what dataset to use as the evaluation set
    parser.add_argument('-test_source', type=str, default='h36m')
    # whether to use pre-augmented training data
    parser.add_argument('-pre_aug', type=bool, default=False)
    # the path to the pre-augmented dataset
    parser.add_argument('-pre_aug_dir', type=str, default='../data/augmented_evo_10.npy')
    # the path of pre-trained check-point
    parser.add_argument('-ckpt_dir', type=str)
    ##-----------------------------------------------------------------------##    
    ## dataset settings
    # whether to only predict 14 joints
    parser.add_argument('-pred14', type=bool, default=False)   
    # whether to add 3D poses fitted by SMPL model
    parser.add_argument('-SMPL', type=bool, default=False)      
    # perform normalization for each pose instead of all the poses
    parser.add_argument('-norm_single', type=bool, default=False) 
    # how much weight is given to the new poses of SMPL
    parser.add_argument('-SMPL_weight', type=float, default=0.5)   
    # whether to change the image size of the cameras
    parser.add_argument('-change_size', type=bool, default=False)
    # virtual image size if changed
    parser.add_argument('-vir_img_size', type=int, default=256)
    # use only a subset of training examples for weakly-supervised experiments
    parser.add_argument('-ws', type=bool, default=False)
    # the path to evolved training examples for weakly-supervised experiments
    parser.add_argument('-evolved_path', type=str, default=None)
    # the training sample used if no path is provided
    parser.add_argument('-ws_name', type=str, default='S1')
    # whether to visualize the dataset
    parser.add_argument('-visualize', type=bool, default=False)
    # whether to show the ambiguous pairs in the dataset
    parser.add_argument('-show_ambi', type=bool, default=False)    
    ##-----------------------------------------------------------------------##    
    # Optimizer settings
    parser.add_argument('-optim_type', type=str, default='adam')
    parser.add_argument('-lr', type=float, default=0.001, help="sgd: 0.5, adam: 0.001")
    parser.add_argument('-weight_decay', type=float, default=0.0)
    parser.add_argument('-momentum', type=float, default=0.9, help="sgd: 0.9")
    # reduce the learning rate after each milestone
    #parser.add_argument('-milestones', type=list, default=[6, 12, 18])
    parser.add_argument('-milestones', type=list, default=[50, 100, 150])
    # how much to reduce the learning rate
    parser.add_argument('-gamma', type=float, default=1)
    ##-----------------------------------------------------------------------##  
    ## usage configuration
    # whether to train a model or deploy a trained model
    parser.add_argument('-train', type=bool, default=False)
    parser.add_argument('-evaluate', type=bool, default=False)
    # evaluate a batch of models
    parser.add_argument('-evaluate_batch', type=bool, default=False)
    # whether to save the trained model
    parser.add_argument('-save', type=bool, default=True)    
    # evaluate for each action
    parser.add_argument('-evaluate_action', type=bool, default=True)
    parser.add_argument('-produce', type=bool, default=False)
    opt = parser.parse_args()
    return opt