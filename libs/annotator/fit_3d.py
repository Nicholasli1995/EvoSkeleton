"""
Obtain 3D skeleton with 2D key-points as inputs using SMPLify
Please run this script in Python 2 environment for now.
TODO: transfer this tool to Python 3.
"""

from smpl_webuser.serialization import load_model 
from smplify.fit_3d import run_single_fit
# you can use multi-processing to fit the images in parallel
#from multiprocessing import Pool 

import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import cv2
import os 

# whether to use the regularization terms
sph_regs = None
# number of blend shapes to use
n_betas = 1
# focal length of the camera
flength = 5000
pix_thsh = 25
scale_factor = 1
# viewpoints for rendering
do_degrees = [0.]

def main(opt):
    model = load_model(opt.model_dir)
    annotation_path = os.path.join(opt.dataset_dir, 'annotation.npy')
    assert os.path.exists(annotation_path), "Please prepare the 2D annotation first."
    annotation = np.load(annotation_path, allow_pickle=True).item()
    if opt.save_image and not os.path.exists(os.path.join(opt.dataset_dir, 'fitted')):
        os.makedirs(os.path.join(opt.dataset_dir, 'fitted'))
    for (image_name, annots) in annotation.items():
        assert 'p2d' in annots, "The image must be annotated with 2D key-points"
        joints_2d = annots['p2d']
        # use 14 key-points for model fitting
        # one may adjust this number by considering different 2D-3D corespondence
        if joints_2d.shape[0] > 14:
            joints_2d = joints_2d[:14, :]        
        # Prepare fitting parameters
        filling_list = range(12) + [13]
        conf = np.zeros(joints_2d.shape[0])
        conf[filling_list] = 1.0

        img = cv2.imread(os.path.join(opt.dataset_dir, image_name))
        
        # Run single fit
        params, vis = run_single_fit(
            img,
            joints_2d, 
            conf, 
            model, 
            regs=sph_regs,
            n_betas=n_betas,
            flength=flength,
            pix_thsh=pix_thsh,
            scale_factor=scale_factor,
            viz=opt.viz,
            do_degrees=do_degrees
        )

        # Show result then close after 1 second
        f = plt.figure(figsize=(6, 3))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for di, deg in enumerate(do_degrees):
            plt.subplot(122)
            plt.cla()
            plt.imshow(vis[di])
            plt.draw()
            plt.pause(1)

        # record the params
        annotation[image_name]['fitting_params'] = params
        # save fitted image
        if opt.save_image:
            image_name = image_name.replace(".jpg", ".png")
            img_save_path = os.path.join(opt.dataset_dir, 'fitted', image_name)
            f.savefig(img_save_path)   
            plt.close(f)
            print("fitted image saved at ", img_save_path)
            
    # save the annotation dictionary with fitted parameters
    np.save(os.path.join(opt.dataset_dir, "fitted.npy"), annotation)
    print('3D prameters saved at ' + os.path.join(opt.dataset_dir, "fitted.npy"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Annotation')
    parser.add_argument('-dataset_dir', type=str)
    parser.add_argument('-model_dir', type=str)
    parser.add_argument('-save_image',default=True, type=bool)
    # visualize intermeadiate results
    parser.add_argument('-viz',default=False, type=bool)
    opt = parser.parse_args() 
    main(opt)