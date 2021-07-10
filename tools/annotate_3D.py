"""
Interactive annotation tool for 3D human pose estimation.
Given an image and a coarse 3D skeleton estimation, the user can interactively
modify the 3D parameters and save them as the ground truth.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import imageio
import sys
import os

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from cv2 import projectPoints

sys.path.append("../")

from libs.skeleton.anglelimits import get_basis1, normalize, gram_schmidt_columns
from libs.skeleton.anglelimits import nt_parent_indices, nt_child_indices, di_indices
from libs.skeleton.anglelimits import get_normal, di, a, to_spherical, to_xyz, bone_name

''' GLOBAL VARIABLES '''
angle_idx = 0 # Bone angle to adjust
direction = 0 # Direction to rotate, (0 - x, 1 - y, 2 - z) for upper arm only
step = 3 # 3 degrees for step size
step_radian = step * np.pi / 180
local_system_map = {1:0, 3:0, 5:1, 7:1, 2:2, 4:3, 6:4, 8:5}
line_index_map = {1:11, 3:14, 5:4, 7:1, 2:12, 4:15, 6:5, 8:2}
parent_indices = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
child_indices = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
direction_name = ['x', 'y', 'z']

# translation vector of the camera
t = None
# focal length of the camera
f = None
# intrinsic matrix for camera projection
intrinsic_mat = None

# Objects for ploting
fig = None
plot_ax = None
img_ax = None
skeleton = None
lines = None 
points = None
RADIUS = 1 # Space around the subject

# hierarchical representation
local_systems = None
need_to_update_lc = False
bones_global = None
bones_local = None
angles = None

# file path
annotation_path = None
annotation = None
img_name = None

# some joint correspondence
index_list = [13, 14, 129, 145]
H36M_IDS = [0, 2, 5, 8, 1, 4, 7, 3, 12, 15, 24, 16, 18, 20, 17, 19, 21]
USE_DIMS = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# keyboard inputs
bone_idx_keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
global_rot_key ='0'
inc_step_key = 'd'
dec_step_key = 'f'
ang_inc_key = 'up'
ang_dec_key = 'down'
ang_cw_key = 'right'
ang_ccw_key = 'left'
save_key = 'm'

def press(event):
    """
    Call-back function when user press any key.
    """
    global angle_idx, direction, need_to_update_lc
    global bones_global, bones_local, skeleton, angles, local_systems

    if event.key == 'p':
        plot_ax.plot([np.random.rand()], [np.random.rand()], [np.random.rand()], 'ro')
        fig.canvas.draw()

    if event.key in bone_idx_keys:  angle_idx = int(event.key) - 1 
    if event.key == global_rot_key: angle_idx = None
    if event.key == inc_step_key:   direction = (direction + 1) % 3
    if event.key == dec_step_key:   direction = (direction - 1) % 3

    if event.key == ang_inc_key or event.key == ang_dec_key:
        update_skeleton(angle_idx, event.key)

    if event.key == ang_cw_key or event.key == ang_ccw_key:
        if angle_idx in [2, 4, 6, 8]:
            update_skeleton(angle_idx, event.key)

    if event.key == save_key:
        save_skeleton()

    if angle_idx is not None:
        notes = 'current limb: ' + bone_name[angle_idx + 1]
        # update local coordinate systems if needed
        if need_to_update_lc:
            # compute the local coordinate system
            bones_global, bones_local, local_systems = to_local(skeleton)
            # convert the local coordinates into spherical coordinates
            angles = to_spherical(bones_local)
            angles[:,1:] *= 180/np.pi
            # need to update local coordinate system once after global rotation
            need_to_update_lc = False            
    else:
        notes = 'global rotation: '

    if angle_idx in [None, 1, 3, 5, 7]:
        notes += ' direction: ' + direction_name[direction]
    if event.key not in ['up', 'down', 'right', 'left']:
        print(notes)
    plot_ax.set_xlabel(notes)
    fig.canvas.draw_idle()
        
def show3Dpose(channels, 
               ax, 
               lcolor="#3498db", 
               rcolor="#e74c3c", 
               add_labels=True,
               gt=False,
               pred=False,
               inv_z=False
               ): 

    vals = np.reshape( channels, (32, -1) )

    I   = parent_indices
    J   = child_indices
    LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    
    lines = []

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        line = ax.plot(x,y, z,  lw=2, c=lcolor if LR[i] else rcolor)
        lines.append(line)

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]

    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
    ax.set_aspect('auto')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

    if inv_z:
        ax.invert_zaxis() 

    return lines

def to_local(skeleton):
    """
    Convert bone vector in skeleton format to local coordinate system
    """
    global local_systems
    
    v1, v2, v3 = get_basis1(skeleton)

    # Compute vector of left hip to right hip
    left_hip = skeleton[6]
    right_hip = skeleton[1]
    v4 = normalize(right_hip - left_hip)

    # v5 is the cross product of v4 and v2 (front-facing vector for lower-body)
    v5 = normalize(np.cross(v4, v2))

    # Compute orthogonal coordinate systems using GramSchmidt
    # Make sure the directions roughly align
    # For upper body, we use v1, v2 and v3
    # For lower body, we use v4, v2 and v5
    system1 = gram_schmidt_columns(np.hstack([v1.reshape(3,1), 
                                              v2.reshape(3,1), 
                                              v3.reshape(3,1)]))

    system2 = gram_schmidt_columns(np.hstack([v4.reshape(3,1), 
                                              v2.reshape(3,1), 
                                              v5.reshape(3,1)]))

    local_systems = [system1, system2]
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

        bone = bones[bone_idx]
        if bone_idx in [0, 1, 3, 5, 7]:
            # Bones directly connected to torso
            # Upper body - 0, 1, 3
            # Lower body - 5, 7
            if bone_idx in [0, 1, 3]:   bones_local[bone_idx] = system1.T @ bone
            else:                       bones_local[bone_idx] = system2.T @ bone
        else:
            if bone_idx in [2, 4]:  parent_R = system1
            else:                   parent_R = system2

            # parent bone index is smaller than 1
            vector_u = normalize(bones[bone_idx - 1])
            di_index = di_indices[bone_idx]

            vector_v, flag = get_normal(parent_R@di[:, di_index],
                                  parent_R@a,
                                  vector_u)
            vector_w = np.cross(vector_u, vector_v)

            local_system = gram_schmidt_columns(np.hstack([vector_u.reshape(3,1), 
                                              vector_v.reshape(3,1), 
                                              vector_w.reshape(3,1)]))

            local_systems.append(local_system)
            bones_local[bone_idx] = local_system.T @ bone

    return bones, bones_local, local_systems

def update_line(line_idx, parent_idx, child_idx):
    """
    Update 3D line segments.
    """
    global lines

    # update 3D lines
    parent, child = skeleton[parent_idx], skeleton[child_idx]

    x = np.array([parent[0], child[0]])
    y = np.array([parent[1], child[1]])
    z = np.array([parent[2], child[2]])

    lines[line_idx][0].set_data(x, y)
    lines[line_idx][0].set_3d_properties(z)

    fig.canvas.draw_idle()

def update_global(angle_idx):
    """
    Update bone vectors.
    """
    global bones_global, bones_local, local_systems, skeleton
    bones_global[angle_idx] = local_systems[local_system_map[angle_idx]] @ bones_local[angle_idx]
    skeleton[nt_child_indices[angle_idx]] = skeleton[nt_parent_indices[angle_idx]] \
                                            - bones_global[angle_idx]  

    line_idx = line_index_map[angle_idx]
    parent_idx = nt_parent_indices[angle_idx]
    child_idx = nt_child_indices[angle_idx]

    update_line(line_idx, parent_idx, child_idx)

def rotate_global(rot):
    """
    Change the global orientation of the 3D skeleton.
    """
    global skeleton
    hip = skeleton[0].reshape(1,3)
    temp_skeleton = skeleton - hip
    skeleton = (rot.as_matrix() @ temp_skeleton.T).T + hip

    for line_idx in range(len(parent_indices)):
        update_line(line_idx, 
                    parent_indices[line_idx], 
                    child_indices[line_idx]
                    )

def update_skeleton(angle_idx, key_name):
    """
    Update the 3D skeleton with a specified keyboard input.
    """
    global need_to_update_lc, local_systems

    # Rotate the lower-limb
    if angle_idx in [2, 4, 6, 8]:
        if key_name == 'up':        angles[angle_idx, 1] += step
        elif key_name == 'down':    angles[angle_idx, 1] -= step
        elif key_name == 'left':    angles[angle_idx, 2] += step
        elif key_name == 'right':   angles[angle_idx, 2] -= step 
        
        temp = angles[angle_idx].copy()
        temp[1:] *= np.pi / 180
        bones_local[angle_idx] = to_xyz(temp.reshape(1,3))
        
        update_global(angle_idx)

    # Rotate the upper-limb with respect to the torso coordinate system
    if angle_idx in [1, 3, 5, 7]:

        # Local rotation vector
        rot_vec = np.array([0., 0., 0.])
        rot_vec[direction] = 1. if key_name == 'up' else -1.
        rot = R.from_rotvec(rot_vec*step_radian)
        bones_local[angle_idx] = rot.apply(bones_local[angle_idx]) 

        # Global rotation vector
        rot_vec2 = local_systems[local_system_map[angle_idx]][:, direction].copy()
        rot_vec2 *= 1. if key_name == 'up' else -1.
        rot2 = R.from_rotvec(rot_vec2*step_radian)

        # Local rotation vector for child/lower limb
        temp = local_systems[local_system_map[angle_idx + 1]]
        local_systems[local_system_map[angle_idx + 1]] = rot2.as_matrix() @ temp

        # update parent and child bone
        update_global(angle_idx)
        update_global(angle_idx + 1)

    # Global rotation
    if angle_idx is None and key_name in ['up', 'down']:
        need_to_update_lc = True
        rot_vec = np.array([0., 0., 0.])
        rot_vec[direction] = 1. if key_name == 'up' else -1.
        rot = R.from_rotvec(rot_vec*step_radian)
        rotate_global(rot)

    # Update the 2D Projection
    update_projection()

def update_projection():
    """
    Update the 2D projection of the 3D key-points.
    """
    global points
    points.pop(0).remove()
    proj2d = projectPoints(skeleton, 
                           np.zeros((3)), 
                           t, 
                           intrinsic_mat, 
                           np.zeros((5))
                           )
    proj2d = proj2d[0].reshape(-1,2)
    points = img_ax.plot(proj2d[:,0], proj2d[:,1], 'ro')   
    fig.canvas.draw_idle()

def save_skeleton():
    """
    Save the annotation file.
    """
    global annotation
    annotation[img_name]['p3d'] = skeleton
    np.save(annotation_path, annotation)
    print('Annotated 3D parameters saved at ' + annotation_path)

def visualize(pose, skeleton, img):
    """
    Initialize the 3D and 2D plots.
    """
    global lines, points, fig, plot_ax, img_ax, intrinsic_mat
    fig = plt.figure() 
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    
    # 3D pose plot
    plot_ax = plt.subplot(121, projection='3d')

    lines = show3Dpose(pose, plot_ax)
    fig.canvas.mpl_connect('key_press_event', press)
    plot_ax.set_title('1-9: limb selection, 0: global rotation, arrow keys: rotation')
    # Image plot
    img_ax = plt.subplot(122)
    img_ax.imshow(img)
    intrinsic_mat = np.array([[f[0], 0.00e+00, float(img.shape[1])/2],
                              [0.00e+00, f[1], float(img.shape[0])/2],
                              [0.00e+00, 0.00e+00, 1.00e+00]])
    proj2d = projectPoints(skeleton, 
                           np.zeros((3)), 
                           t, 
                           intrinsic_mat, 
                           np.zeros((5))
                           )
    proj2d = proj2d[0].reshape(-1,2)
    points = img_ax.plot(proj2d[:,0], proj2d[:,1], 'ro')
    # Show the plot
    plt.show()

def create_python3_file(opt):
    """
    The fitted parameters are stored using python 2. 
    Create a Python 3 file from it when this script is executed for the first time.
    """
    fitted_path = os.path.join(opt.dataset_dir, "fitted.npy")
    annotation_path = os.path.join(opt.dataset_dir, "annot_3d.npy")
    if not os.path.exists(annotation_path):
    # The fitting parameters are obtianed in Python 2 environment,
    # thus the encoding argument is used here        
        fitted = np.load(fitted_path, encoding='latin1', allow_pickle=True).item()   
        np.save(annotation_path, fitted)
    return

def main(opt):
    global t, f, angles, bones_global, bones_local, need_to_update_lc, \
    local_system, skeleton, annotation_path, img_name, annotation
    create_python3_file(opt)
    annotation_path = os.path.join(opt.dataset_dir, "annot_3d.npy")
    annotation = np.load(annotation_path, allow_pickle=True).item()    
    for img_name in annotation.keys():
        # e.g., img_name = '260.jpg'
        img_name = '260.jpg'
        # select one unannotated image and start the interactive annotation
        if 'p3d' in annotation[img_name]:
            continue
        fitting_params = annotation[img_name]['fitting_params']
        img_path = os.path.join(opt.dataset_dir, img_name)
        
        img = imageio.imread(img_path)

        # Convert smpl format to Human 3.6M Format
        skeleton_smpl = fitting_params['v'].reshape(-1, 3)
        skeleton = np.zeros((32, 3))
        skeleton[USE_DIMS] = skeleton_smpl[H36M_IDS]

        pose = skeleton.reshape(-1)

        t = fitting_params['cam_t']
        f = fitting_params['f']

        # Convert skeleton to local coordinate system
        bones_global, bones_local, local_system = to_local(skeleton)

        # Convert the local coordinates to spherical coordinates
        angles = to_spherical(bones_local)
        angles[:, 1:] *= 180 / np.pi

        # Set update local coordinate flag only after global rotation
        need_to_update_lc = False

        # Visualize
        visualize(pose, skeleton, img)
        
        # annotate only one image at a time
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Interactive Tool')
    parser.add_argument('-d', '--dataset_dir', default=None, type=str)
    opt = parser.parse_args()
    main(opt)