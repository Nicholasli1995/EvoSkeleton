# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Shichao Li (nicholas.li@connect.ust.hk)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from libs.hhr.core.inference import get_max_preds, get_max_preds_soft
from libs.hhr.core.loss import get_max_preds_soft_pt

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

from libs.hhr.utils.transforms import get_affine_transform
from libs.hhr.utils.transforms import affine_transform_modified

def get_distance(gt, pred):
    # gt: [n_joints, 2]
    # pred: [n_joints, 2]
    sqerr = (gt - pred)**2
    sqerr = sqerr.sum(axis = 1, keepdims=True)
    dist = np.sqrt(sqerr)
    return dist

def accuracy_pixel(output, meta_data, image_size = (288.0, 384.0), arg_max='hard'):
    ''' 
    Report errors in terms of pixels in the original image plane.
    '''
    if arg_max == 'soft':
        if isinstance(output, np.ndarray):
            pred, max_vals = get_max_preds_soft(output)
        else:
            pred, max_vals = get_max_preds_soft_pt(output)
    elif arg_max == 'hard':
        if not isinstance(output, np.ndarray):
            output = output.data.cpu().numpy()
        pred, max_vals = get_max_preds(output)
    else:
        raise NotImplementedError
        
    # multiply by down-sample ratio
    if not isinstance(pred, np.ndarray):
        pred = pred.data.cpu().numpy()
        max_vals = max_vals.data.cpu().numpy()
    pred *= image_size[0]/output.shape[3]
    # inverse transform and compare pixel didstance
    centers, scales, rots = meta_data['center'], meta_data['scale'], meta_data['rotation']
    centers = centers.data.cpu().numpy()
    scales = scales.data.cpu().numpy()
    rots = rots.data.cpu().numpy()
    joints_original_batch = meta_data['j_original'].data.cpu().numpy()

    distance_list = []
    all_src_coordinates = []
    for sample_idx in range(len(pred)):
        trans_inv = get_affine_transform(centers[sample_idx], scales[sample_idx], rots[sample_idx], image_size, inv=1)
        joints_original = joints_original_batch[sample_idx]        
        pred_src_coordinates = affine_transform_modified(pred[sample_idx], trans_inv) 
        all_src_coordinates.append(pred_src_coordinates.reshape(1, len(pred_src_coordinates), 2))
        distance_list.append(get_distance(joints_original, pred_src_coordinates))
    all_distance = np.hstack(distance_list)
    acc = all_distance
    avg_acc = all_distance.mean()
    cnt = len(distance_list) * len(all_distance)
    return acc, avg_acc, cnt, np.concatenate(all_src_coordinates, axis=0), pred, max_vals
