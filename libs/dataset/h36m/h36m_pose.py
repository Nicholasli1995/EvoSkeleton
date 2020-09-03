#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Shichao Li (nicholas.li@connect.ust.hk)
# ------------------------------------------------------------------------------
import numpy as np
import copy
import torch
import cv2
import random

from libs.dataset.h36m.pose_dataset import JointsDataset
from libs.hhr.utils.transforms import get_affine_transform
from libs.hhr.utils.transforms import affine_transform
from libs.hhr.utils.transforms import fliplr_joints

import logging

logger = logging.getLogger(__name__)


## Human 3.6M dataset class
class H36MDataset(JointsDataset):
    '''
    COCO annotation:
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    H36M annotation:
        H36M_NAMES[0]  = 'Hip'
        H36M_NAMES[1]  = 'RHip'
        H36M_NAMES[2]  = 'RKnee'
        H36M_NAMES[3]  = 'RFoot'
        H36M_NAMES[4]  = 'LHip'
        H36M_NAMES[5]  = 'LKnee'
        H36M_NAMES[6]  = 'LFoot'
        H36M_NAMES[7] = 'Spine'
        H36M_NAMES[8] = 'Thorax'
        H36M_NAMES[9] = 'Neck/Nose'
        H36M_NAMES[10] = 'Head'
        H36M_NAMES[11] = 'LShoulder'
        H36M_NAMES[12] = 'LElbow'
        H36M_NAMES[13] = 'LWrist'
        H36M_NAMES[14] = 'RShoulder'
        H36M_NAMES[15] = 'RElbow'
        H36M_NAMES[16] = 'RWrist'   
    "skeleton": [
        [0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8],
        [8,9], [9,10], [8,11], [11,12], [12,13], [8,14], [14,15], [15,16]] 
    permutation from H36M to COCO:
    [9, 7, 8, 0, 10, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
    permutation to get back:
        
    '''

    def __init__(self, cfg, is_train, annot_path, transform=None):
        super().__init__(cfg, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.loss_type = cfg.MODEL.TARGET_TYPE
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        # path to pre-processed annotation
        self.annot_path = annot_path
        self.num_joints = 17
        self.flip_pairs = [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (3, 11, 12, 13, 14, 15, 16)
        self.joints_weight = np.ones((self.num_joints,1), np.float32)
        self.joints_weight[[7,8,13,14]] = 1.2
        self.joints_weight[[9,10,15,16]] = 1.5
        
        ## permute joint order for fine-tuning purpose
        self.fine_tune_re_order = [9, 7, 8, 0, 10, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
        
        self.ratio = float(cfg.MODEL.IMAGE_SIZE[0]/cfg.MODEL.HEATMAP_SIZE[0])

        self.db = self._get_db()
        logging.info('=> total annotation for images: {}'.format(len(self.db)))
        
        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)
            
        logging.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        gt_db = np.load(self.annot_path)
        # permute joints
        for record in gt_db:
            record['p_2d'] = record['p_2d'][self.fine_tune_re_order, :]
        return gt_db
    
    def get_weights(self):
        weights = []
        for sample_idx in range(len(self.db)):
            path = self.db[sample_idx]['path']
            if 'S6' in path or 'S8' in path:
                weights.append(1.5)
            else:
                weights.append(1.0)
        return weights
    
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)

        return center, scale
    
    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        image_file = db_rec['path']


#        data_numpy = cv2.imread(
#            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
#        )

        # opencv 3
        data_numpy = cv2.imread(
            image_file, 1 | 128
        )        

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['p_2d']
        joints_original = joints.copy()
        joints_vis = np.ones(joints.shape, dtype=np.float32)
        c, s = self._xywh2cs(0, 0, data_numpy.shape[1], data_numpy.shape[0])
        score = 1
        r = 0

        if self.is_train:
            # do not do half body transform since there is not so much occlusion
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.3 else 0
            
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                # set joints to in-visible if they are out-side of the image
                if joints[i, 0] >= self.image_width or joints[i, 1] >= self.image_height:
                    joints_vis[i, 0] = 0.0
        
        
        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'joints': joints,
            'joints_vis': joints_vis,
            'j_original':joints_original, #original coordinates
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'trans':trans,
            'bbox':[0, 0, data_numpy.shape[1], data_numpy.shape[0]]
        }

        return input, target, target_weight, meta    
    
    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type in ['gaussian', 'coordinate'], \
            'Unsupported target type'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        elif self.target_type == 'coordinate':
            target = joints/self.ratio  
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight    