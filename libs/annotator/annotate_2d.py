"""
Annotate 2D key-points.
"""
from glob import glob
from os.path import join
import imageio
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import os

''' ANNOTATION CONTROLS '''
names = ['ra',      # 0 right ankle
         'rk',      # 1 right knee2
         'rh',      # 2 right hip
         'lh',      # 3 left hip
         'lk',      # 4 left knee
         'la',      # 5 left ankle
         'rw',      # 6 right wrist
         're',      # 7 right elbow
         'rs',      # 8 right shoulder
         'ls',      # 9 left shoulder
         'le',      # 10 left elbow
         'lw',      # 11 left wrist
         'ne',      # 12 neck
         'ht',      # 13 head top
         'sp',      # 14 spine
         'th',      # 15 thorax
         'ns']      # 16 nose

I = np.array([1,2,3,4,5,6,7,8,9,10,11,13]) - 1 # Start points
J = np.array([2,3,4,5,6,7,8,9,10,11,12,14]) - 1 # End points

joints = np.array([]).reshape(0, 2)
fig = None
cid = None
plots = None
n = len(names)
joint_shape = (n, 2)

def onpick(event):
    global joints, fig, plots

    if event.button == 1:
        if len(joints) < n:
            ind = len(joints)
            joint = np.array([event.xdata, event.ydata])
            print(joint.shape, joints.shape)
            joints = np.vstack((joints, joint))

            plots[0].remove()
            plots = plt.plot(joints[:, 0], joints[:, 1], 'ro')
            fig.canvas.draw()
            
            print(names[ind] + ": " + str(joint))

def onkey(event):
    global joints, fig, cid, plots

    if event.key == 'q':
        fig.canvas.mpl_disconnect(cid)
        return joints

    if event.key == 'c': 
        joints = np.array([]).reshape(0, 2)
        plots[0].remove()
        plots = plt.plot([], [], 'ro')
        fig.canvas.draw()
        
''' VIEW '''
def main(args):
    global joints, fig, cid, plots
    img_name_list = []

    ''' SELECT DATASET '''
    # Load LSD Dataset
    if args.dataset == 'lsd':
        dataset_path = './../dataset/lsd'
        
        txt_path = os.path.join(dataset_path, 'cases.txt')
        img_dir = os.path.join(dataset_path, 'images')
        joints_path = os.path.join(dataset_path, 'annotations/est.npy')

        with open(txt_path) as f:
            content = f.read()
            
            for num in content.split('\r'):
                if num == '':
                    continue

                img_name = os.path.join(img_dir, 'im' + num.zfill(4) + '.jpg')
                img_name_list.append(img_name)

    # Load Custom Dataset
    else:
        dataset_path = args.dataset 
        img_dir = os.path.join(dataset_path, 'images')
        joints_path = os.path.join(dataset_path, 'annotations/est.npy')

        img_name_list = sorted(glob(os.path.join(img_dir, '*.jpg')))

    ''' LOAD IMAGES AND JOINTS '''
    joints_list = np.array([]).reshape((0, joint_shape[0], joint_shape[1]))
    if os.path.exists(joints_path):
        joints_list = np.load(joints_path)

    ''' ANNOTATE 2D JOINTS '''
    for i, img_name in enumerate(img_name_list):
        if len(joints_list) > i:
            joints = joints_list[i]
        else:
            joints = np.array([]).reshape(0, 2)

        img = np.array(imageio.imread(img_name))

        fig = plt.gcf() 
        ax = plt.gca() 
        ax.imshow(img)
        fig.canvas.mpl_connect('button_press_event', onpick)
        cid = fig.canvas.mpl_connect('key_press_event', onkey)

        if joints == []:
            plots = ax.plot([], [], 'ro')
        else:
            plots = ax.plot(joints[:, 0], joints[:, 1], 'ro')
        plt.show()

        if joints == []:
            joints = np.zeros(joint_shape)

        if len(joints_list) > i:
            joints_list[i] = joints
        else:
            joints_list = np.vstack((joints_list, joints[None]))
        print(joints_list[i])

    joints_list = np.array(joints_list)

    ''' SAVE JOINTS '''
    if args.save:
        np.save(joints_path, joints_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Annotation')

    parser.add_argument('-d', '--dataset_dir', default=None, type=str)
    parser.add_argument('-s', '--save', default=True, type=bool)

    args = parser.parse_args()

    main(args)