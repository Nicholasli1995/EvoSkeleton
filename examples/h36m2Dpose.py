"""
Examplar code showing how to use pre-trained heatmap regression model H() to 
perform 2D pose estimation on Human 3.6M images. 
"""

import sys
sys.path.append("../")

from libs.hhr.config import cfg
from libs.hhr.config import update_config
from libs.hhr.utils.utils import get_model_summary
from libs.model.pose_hrnet import get_pose_net
from libs.hhr.utils.transforms import get_affine_transform
from libs.hhr.core.loss import get_max_preds_soft_pt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import argparse
import os
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

pose_connection = np.array([[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8],
                            [8,9], [9,10], [8,11], [11,12], [12,13], [8,14], [14,15],
                            [15,16]], dtype=np.int
                           )
I = pose_connection[:, 0]
J = pose_connection[:, 1]
pose_color = np.array([[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
                       [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], 
                       [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
                       [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255]
                       ])/255.
re_order = [3, 12, 14, 16, 11, 13, 15, 1, 2, 0, 4, 5, 7, 9, 6, 8, 10]

def show2Dpose(vals, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
    for i in np.arange( len(I) ):
        x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
        ax.plot(x, y, c=pose_color[i])
    
def parse_args():
    parser = argparse.ArgumentParser(description='2D pose estimation example')
    parser.add_argument('--cfg',
                        help='configuration file',
                        default='./h36m2Dpose/cfgs.yaml',
                        type=str)
    parser.add_argument('--data_path',
                        help='path to pre-processed testing images',
                        default='./h36m2Dpose/cropped',
                        type=str)
    args = parser.parse_args()
    return args


def xywh2cs(x, y, w, h, aspect_ratio=0.75):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / 200, h * 1.0 / 200], dtype=np.float32)

    return center, scale
    
def gather_inputs(args, logger, image_size = (288, 384)):
    root = args.data_path
    img_names = os.listdir(root)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )    
    transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])    
    inputs = []
    # these testing images were cropped from videos of subject 9 and 11
    for name in img_names:
        pass
        image_file = os.path.join(root, name)
        data_numpy = cv2.imread(image_file, 1 | 128)        
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        c, s = xywh2cs(0, 0, data_numpy.shape[1], data_numpy.shape[0])
        r = 0    
        trans = get_affine_transform(c, s, r, image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (image_size[0], image_size[1]),
            flags=cv2.INTER_LINEAR)
        inputs.append(transform(input).unsqueeze(0))
    return torch.cat(inputs)

def unnormalize(tensor):
    img = tensor.data.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = np.transpose((((img * std) + mean) * 255).astype(np.uint8), (1, 2, 0))
    return img

def visualize(inputs, model):
    output = model(inputs)
    pred, max_vals = get_max_preds_soft_pt(output)
    pred = pred.data.cpu().numpy()
    pred = pred[:, re_order, :]
    plt.figure()
    for i in range(len(pred)):
        ax = plt.subplot(1, len(pred), i+1)
        ax.imshow(unnormalize(inputs[i]))
        ax.plot(pred[i][:, 0], pred[i][:, 1], 'ro')
        show2Dpose(pred[i], ax)
    return

def main():
    args = parse_args()
    update_config(cfg, args)

    logging.basicConfig(format='%(asctime)-15s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = get_pose_net(cfg, is_train=False)
    
    # load the pre-trained weights
    if not os.path.exists('./h36m2Dpose/final_state.pth'):
        logger.info('Please download the pre-trained model first.')
        return
    checkpoint = torch.load('./h36m2Dpose/final_state.pth')
    model.load_state_dict(checkpoint)
    
    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))
    
    # modify the configuration file for multiple GPUs
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda() 
    
    inputs = gather_inputs(args, logger)
    visualize(inputs, model)

if __name__ == '__main__':
    main()