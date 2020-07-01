"""
3D pose estimation based on 2D key-point coordinates as inputs.
Author: Shichao Li
Email: nicholas.li@connect.ust.hk
"""

import logging
import os
import sys
sys.path.append("../")

import torch
import numpy as np

import libs.parser.parse as parse
import libs.utils.utils as utils
import libs.dataset.h36m.data_utils as data_utils
import libs.trainer.trainer as trainer

def main():
    # logging configuration
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s]: %(message)s"
                        )        
    
    # parse command line input
    opt = parse.parse_arg()
    
    # Set GPU
    opt.cuda = opt.gpuid >= 0
    if opt.cuda:
        torch.cuda.set_device(opt.gpuid)
    else:
        logging.info("GPU is disabled.")

    # dataset preparation
    train_dataset, eval_dataset, stats, action_eval_list = \
    data_utils.prepare_dataset(opt)
    
    if opt.train:
        # train a cascaded 2D-to-3D pose estimation model
        record = trainer.train_cascade(train_dataset, 
                                       eval_dataset, 
                                       stats, 
                                       action_eval_list, 
                                       opt
                                       )
        utils.save_ckpt(opt, record, stats)
        
    if opt.visualize:
        # visualize the inference results of a pre-trained model
        cascade = torch.load(opt.ckpt_path)        
        if opt.cuda:
            cascade.cuda()  
        utils.visualize_cascade(eval_dataset, cascade, stats, opt)
    if opt.evaluate:
        # evalaute a pre-trained cascade
        cascade, stats = utils.load_ckpt(opt)
        trainer.evaluate_cascade(cascade, 
                                 eval_dataset, 
                                 stats, 
                                 opt, 
                                 action_wise=opt.eval_action_wise,
                                 action_eval_list=action_eval_list
                                 ) 
        
if __name__ == "__main__":
    main()