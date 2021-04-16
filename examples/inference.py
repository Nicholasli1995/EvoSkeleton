"""
Am examplar script showing inference on the newly collected images in U3DPW. 
"""

import sys
sys.path.append("../")

import libs.model.model as libm
from libs.dataset.h36m.data_utils import unNormalizeData

import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt

num_joints = 16
gt_3d = False  
pose_connection = [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8],
                   [8,9], [9,10], [8,11], [11,12], [12,13], [8, 14], [14, 15], [15,16]]
# 16 out of 17 key-points are used as inputs in this examplar model
re_order_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
# paths
data_dic_path = './example_annot.npy'     
model_path = './example_model.th'
stats = np.load('./stats.npy', allow_pickle=True).item()
dim_used_2d = stats['dim_use_2d']
mean_2d = stats['mean_2d']
std_2d = stats['std_2d'] 
# load the checkpoint and statistics
ckpt = torch.load(model_path)
data_dic = np.load(data_dic_path, allow_pickle=True).item()
# initialize the model
cascade = libm.get_cascade()
input_size = 32
output_size = 48
for stage_id in range(2):
    # initialize a single deep learner
    stage_model = libm.get_model(stage_id + 1,
                                 refine_3d=False,
                                 norm_twoD=False, 
                                 num_blocks=2,
                                 input_size=input_size,
                                 output_size=output_size,
                                 linear_size=1024,
                                 dropout=0.5,
                                 leaky=False)
    cascade.append(stage_model)
cascade.load_state_dict(ckpt)
cascade.eval()
# process and show total_to_show examples
count = 0
total_to_show = 10

def draw_skeleton(ax, skeleton, gt=False, add_index=True):
    for segment_idx in range(len(pose_connection)):
        point1_idx = pose_connection[segment_idx][0]
        point2_idx = pose_connection[segment_idx][1]
        point1 = skeleton[point1_idx]
        point2 = skeleton[point2_idx]
        color = 'k' if gt else 'r'
        plt.plot([int(point1[0]),int(point2[0])], 
                 [int(point1[1]),int(point2[1])], 
                 c=color, 
                 linewidth=2)
    if add_index:
        for (idx, re_order_idx) in enumerate(re_order_indices):
            plt.text(skeleton[re_order_idx][0], 
                     skeleton[re_order_idx][1],
                     str(idx+1), 
                     color='b'
                     )
    return

def normalize(skeleton, re_order=None):
    norm_skel = skeleton.copy()
    if re_order is not None:
        norm_skel = norm_skel[re_order].reshape(32)
    norm_skel = norm_skel.reshape(16, 2)
    mean_x = np.mean(norm_skel[:,0])
    std_x = np.std(norm_skel[:,0])
    mean_y = np.mean(norm_skel[:,1])
    std_y = np.std(norm_skel[:,1])
    denominator = (0.5*(std_x + std_y))
    norm_skel[:,0] = (norm_skel[:,0] - mean_x)/denominator
    norm_skel[:,1] = (norm_skel[:,1] - mean_y)/denominator
    norm_skel = norm_skel.reshape(32)         
    return norm_skel

def get_pred(cascade, data):
    """
    Get prediction from a cascaded model
    """
    # forward pass to get prediction for the first stage
    num_stages = len(cascade)
    # for legacy code that does not have the num_blocks attribute
    for i in range(len(cascade)):
        cascade[i].num_blocks = len(cascade[i].res_blocks)
    prediction = cascade[0](data)
    # prediction for later stages
    for stage_idx in range(1, num_stages):
        prediction += cascade[stage_idx](data)
    return prediction

def show3Dpose(channels, 
               ax, 
               lcolor="#3498db", 
               rcolor="#e74c3c", 
               add_labels=True,
               gt=False,
               pred=False
               ):
    vals = np.reshape( channels, (32, -1) )
    I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
    J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
    LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        if gt or pred:
            color = 'k' if gt else 'r'
            ax.plot(x,y, z,  lw=2, c=color)        
        else:
            ax.plot(x,y, z,  lw=2, c=lcolor if LR[i] else rcolor)
    RADIUS = 750 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
    ax.set_aspect('equal')
    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)
    ax.invert_zaxis()
    return

def re_order(skeleton):
    skeleton = skeleton.copy().reshape(-1,3)
    # permute the order of x,y,z axis
    skeleton[:,[0,1,2]] = skeleton[:, [0,2,1]]    
    return skeleton.reshape(96)
    
def plot_3d_ax(ax, 
               elev, 
               azim, 
               pred, 
               title=None
               ):
    ax.view_init(elev=elev, azim=azim)
    show3Dpose(re_order(pred), ax)     
    plt.title(title)    
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

for image_name in data_dic.keys():
    image_path = './imgs/' + image_name
    img = imageio.imread(image_path)
    f = plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(131)
    ax1.imshow(img)
    plt.title('Input image')
    ax2 = plt.subplot(132)
    plt.title('2D key-point inputs: {:d}*2'.format(num_joints))
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    skeleton_pred = None
    skeleton_2d = data_dic[image_name]['p2d']
    # The order for the 2D keypoints is:
    # 'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 
    # 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder'
    # 'RElbow', 'RWrist'
    draw_skeleton(ax2, skeleton_2d, gt=True)
    plt.plot(skeleton_2d[:,0], skeleton_2d[:,1], 'ro', 2)       
    # Nose was not used for this examplar model
    norm_ske_gt = normalize(skeleton_2d, re_order_indices).reshape(1,-1)
    pred = get_pred(cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)))      
    pred = unNormalizeData(pred.data.numpy(),
                           stats['mean_3d'],
                           stats['std_3d'],
                           stats['dim_ignore_3d']
                           )      
    ax3 = plt.subplot(133, projection='3d')
    plot_3d_ax(ax=ax3, 
               pred=pred, 
               elev=10., 
               azim=-90,
               title='3D prediction'
               )    
    adjust_figure(left = 0.05, 
                  right = 0.95, 
                  bottom = 0.08, 
                  top = 0.92,
                  wspace = 0.3, 
                  hspace = 0.3
                  )       
    count += 1       
    if count >= total_to_show:
        break