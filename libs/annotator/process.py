#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from smpl_webuser.serialization import load_model 
from smplify.fit_3d import run_single_fit

from multiprocessing import Pool 

import matplotlib.pyplot as plt 
from glob import glob
import numpy as np 
import argparse
import imageio
import cv2 
import os 

sph_regs = None
n_betas = 1
flength = 5000
pix_thsh = 25
scale_factor = 1
viz = False
do_degrees = [0.]

def main(args):
    img_name_list = []
    param_dir = None

    model = load_model(args.model)

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

                img_name = 'im' + num.zfill(4) + '.jpg'
                img_name = os.path.join(img_dir, img_name)
                img_name_list.append(img_name)

        # Create processed directory
        if not os.path.exists(os.path.join(dataset_path, 'processed')):
            os.mkdir(os.path.join(dataset_path, 'processed'))
        param_dir = os.path.join(dataset_path, 'processed')

    # Load Custom Dataset
    else:
        dataset_path = args.dataset
        img_dir = os.path.join(dataset_path, 'images')
        joints_path = os.path.join(dataset_path, 'annotations/est.npy')

        img_name_list = sorted(glob(os.path.join(img_dir, '*.jpg')))

        if not os.path.exists(os.path.join(dataset_path, 'processed')):
            os.mkdir(os.path.join(dataset_path, 'processed'))
        param_dir = os.path.join(dataset_path, 'processed')

    ''' LOAD IMAGES AND JOINTS '''
    joints_2d = np.load(joints_path)
    if joints_2d.shape[1] > 14:
        joints_2d = joints_2d[:, :14, :]

    ''' PROCESS THE IMAGE AND KEYPOINTS '''
    for i, (joints, img_name) in enumerate(zip(joints_2d, img_name_list)):
        # Prepare variables
        filling_list = range(12) + [13]
        conf = np.zeros(joints.shape[0])
        conf[filling_list] = 1.0

        img = np.array(imageio.imread(img_name))

        # Run single fit
        params, vis = run_single_fit(
            img,
            joints, 
            conf, 
            model, 
            regs=sph_regs,
            n_betas=n_betas,
            flength=flength,
            pix_thsh=pix_thsh,
            scale_factor=scale_factor,
            viz=viz,
            do_degrees=do_degrees
        )

        # Show result then close after 1 second
        f = plt.figure(figsize=(6, 3))
        plt.subplot(121)
        plt.imshow(img)

        for di, deg in enumerate(do_degrees):
            plt.subplot(122)
            plt.cla()
            plt.imshow(vis[di])
            plt.draw()
            plt.pause(1)

        # Save the params
        param_path = os.path.join(param_dir, str(i) + '.npy')
        np.save(param_path, params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Annotation')

    parser.add_argument('-d', '--dataset', default='lsd', type=str)
    parser.add_argument('-m', '--model', type=str)

    args = parser.parse_args() 

    main(args)