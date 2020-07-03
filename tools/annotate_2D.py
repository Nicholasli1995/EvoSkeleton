"""
Annotate 2D key-points interactively.
Press Q to exit the tool.
Press C to remove the annotation.
Press N to go to the next image.
Press Z to save the annotation.
Mouse click to annotate 2D key-points.
"""

import imageio
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import os

from glob import glob

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

joints = np.array([]).reshape(0, 2)
fig = None
ax = None
cid = None
plots = None
img_path_list = None
img_idx = None
annotation = None
annotation_path = None
# whether the image has veen updated or not
updated = False
n = len(names)

def initialization(opt):
    global img_path_list, img_idx, annotation,annotation_path, updated
    img_path_list = sorted(glob(os.path.join(opt.dataset_dir, '*.jpg')))
    img_path_list += sorted(glob(os.path.join(opt.dataset_dir, '*.png')))
    assert len(img_path_list) != 0, "Can not find image files."
    img_idx = -1
    annotation_path = os.path.join(opt.dataset_dir, 'annotation.npy')
    if os.path.exists(annotation_path):
        annotation = np.load(annotation_path).item()
    else:
        annotation = {}    
    return

def plot_image(img_path):
    global plots, updated
    ax.clear()
    img = imageio.imread(img_path)
    ax.imshow(img)
    plots = ax.plot([], [], 'ro')
    plt.show()    
    fig.canvas.draw()
    updated = True
    return

def onpick(event):
    global joints, fig, plots, updated
    
    if event.button == 1:
        if len(joints) < n and updated:
            ind = len(joints)
            joint = np.array([event.xdata, event.ydata])
            joints = np.vstack((joints, joint))
            plots[0].remove()
            plots = plt.plot(joints[:, 0], joints[:, 1], 'ro')
            fig.canvas.draw()
            print(names[ind] + ": " + str(joint))
            if len(joints) == n:
                # record the annotation
                img_name = img_path_list[img_idx].split(os.sep)[-1]
                annotation[img_name] = {'p2d':joints}
                joints = np.array([]).reshape(0, 2)
                updated = False
                print('Please go on to the next image.')

def save_results():
    np.save(annotation_path, annotation)
    print('A Python dictionary has been saved at ' + annotation_path)
    
def onkey(event):
    global joints, fig, cid, plots, img_idx

    if event.key == 'c': 
        # remove the annotation on the image
        joints = np.array([]).reshape(0, 2)
        plots[0].remove()
        plots = plt.plot([], [], 'ro')
        fig.canvas.draw()
        
    if event.key == 'n':
        # go to next image
        if img_idx <= len(img_path_list) - 1:
            # look for the next unannotated image
            img_idx += 1
            while img_path_list[img_idx].split(os.sep)[-1] in annotation:
                img_idx += 1     
            img_idx = len(img_path_list) - 1 if img_idx == len(img_path_list) else img_idx
            plot_image(img_path_list[img_idx])
        else:
            print('Already the last image.')
            save_results()
    
    if event.key == 'z':
        # save the annotation
        save_results()
        
def main(opt):
    global joints, fig, ax, cid, plots, img_idx, img_path_list, annotation
    # show one unannotated image
    for idx in range(len(img_path_list)):
        img_name = img_path_list[idx].split(os.sep)[-1]
        if img_name not in annotation:
            # start with this unannotated image
            img_idx = idx
            break
    if img_idx == -1:
        print('No unannotated image found.')
        return
    fig = plt.gcf() 
    ax = plt.gca() 
    plot_image(img_path_list[img_idx])
    fig.canvas.mpl_connect('button_press_event', onpick)
    cid = fig.canvas.mpl_connect('key_press_event', onkey)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Annotation')
    parser.add_argument('-d', '--dataset_dir', default=None, type=str)
    opt = parser.parse_args()
    initialization(opt)
    main(opt)