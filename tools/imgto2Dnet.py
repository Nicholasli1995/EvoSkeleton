# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Shichao Li (nicholas.li@connect.ust.hk)
# ------------------------------------------------------------------------------

"""
Training and inference of a high-resolution heatmap regression model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import sys
sys.path.append("../")

from libs.hhr.config import cfg
from libs.hhr.config import update_config
from libs.hhr.core.loss import JointsMSELoss, JointsCoordinateLoss, WingLoss
from libs.hhr.core.function import train
from libs.hhr.core.function import validate_pixel
from libs.hhr.utils.utils import get_optimizer
from libs.hhr.utils.utils import save_checkpoint
from libs.hhr.utils.utils import create_logger
from libs.hhr.utils.utils import get_model_summary
from libs.model.pose_hrnet import get_pose_net

import libs.dataset.h36m
from libs.dataset.h36m.h36m_pose import H36MDataset
# run with your configuration file as follows:
# --cfg "./models/experiments/h36m/hrnet/w48_384x288_adam_lr1e-3.yaml" 

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = get_pose_net(cfg, is_train=True)

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    # coordinate loss with soft arg-max
#    criterion = JointsCoordinateLoss(
#        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
#    ).cuda()    
    
    # Wing Loss
#    criterion = WingLoss(
#        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
#    ).cuda()  
    
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    train_dataset = H36MDataset(
        cfg, True, cfg.DATASET.TRAIN_PATH,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = H36MDataset(
        cfg, False, cfg.DATASET.VALID_PATH,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    
    # inference
    # perf_indicator = validate_pixel(
    #     cfg, valid_loader, valid_dataset, model, criterion,
    #     final_output_dir, tb_log_dir, save=True, split='test')    
    # return     
    
    # training
    # train with hard arg-max with MSE loss first and then fine-tune with
    # soft-argmax coordinate loss works well in practice
    #iterations = [3000, 3000, 6000, 3000, 3000]
    # fine-tune with L1 loss
    #iterations = [6000, 3000, 3000]
    iterations = [6000, 6000, 6000, 3000, 3000]

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()
        
        # set total iterations
        if epoch - begin_epoch < len(iterations):
            total_iters = iterations[epoch - begin_epoch]
            logger.info("Total iterations to train: {}".format(total_iters))
        else:
            total_iters = None
            
#       perform validation during training 
#        perf_indicator = validate_pixel(
#            cfg, valid_loader, valid_dataset, model, criterion,
#            final_output_dir, tb_log_dir)
        
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, total_iters=total_iters)

        #perf_indicator = 0.0
        # evaluate on validation set
        perf_indicator = validate_pixel(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir)

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)

if __name__ == '__main__':
    main()
