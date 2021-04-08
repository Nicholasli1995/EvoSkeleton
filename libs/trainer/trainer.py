"""
Utility functions for cascaded model training and evaluation.
"""
import libs.dataset.h36m.data_utils as data_utils
import libs.model.model as model
from libs.utils.utils import compute_similarity_transform

import torch.nn.functional as F
import torch
import numpy as np
import logging

def train_cascade(train_dataset, eval_dataset, stats, action_eval_list, opt):
    """
    Train a cascade of deep neural networks that lift 2D key-points to 3D pose.
    """
    # initialize an empty cascade
    cascade = model.get_cascade()
    stage_record = []
    input_size = len(stats['dim_use_2d'])
    output_size = len(stats['dim_use_3d'])
    # train each deep learner in the cascade sequentially
    for stage_id in range(opt.num_stages):
        # initialize a single deep learner
        stage_model = model.get_model(stage_id + 1,
                                      refine_3d=opt.refine_3d,
                                      norm_twoD=opt.norm_twoD, 
                                      num_blocks=opt.num_blocks,
                                      input_size=input_size,
                                      output_size=output_size,
                                      linear_size=opt.linear_size,
                                      dropout=opt.dropout,
                                      leaky=opt.leaky)
        # record the stage number
        train_dataset.set_stage(stage_id+1)
        eval_dataset.set_stage(stage_id+1)
        for dataset in action_eval_list:
            dataset.set_stage(stage_id+1)
        # move the deep learner to GPU
        if opt.cuda:
            stage_model = stage_model.cuda()
            
        # prepare the optimizer and learning rate scheduler
        optim, sche = model.prepare_optim(stage_model, opt)
        
        # train the model
        record = train(train_dataset, 
                       eval_dataset, 
                       stage_model,
                       optim, 
                       sche, 
                       stats, 
                       action_eval_list, 
                       opt)
        stage_model = record['model']
        # record the training history
        stage_record.append((record['batch_idx'], record['loss']))
        # update current estimates and regression target
        train_dataset.stage_update(stage_model, stats, opt)
        eval_dataset.stage_update(stage_model, stats, opt)
        
        # update evaluation datasets for each action
        if opt.evaluate_action:
            for dataset in action_eval_list:
                dataset.stage_update(stage_model, stats, opt)
        # put the trained model into the cascade
        cascade.append(stage_model.cpu())     
        
        # release memory
        del stage_model    
    return {'cascade':cascade, 'record':stage_record}

def evaluate_cascade(cascade, 
                     eval_dataset, 
                     stats, 
                     opt, 
                     save=False, 
                     save_path=None,
                     action_wise=False, 
                     action_eval_list=None, 
                     apply_dropout=False
                     ):
    """
    Evaluate a cascaded model given a dataset object.
    """
    loss, distance = None, None
    for stage_id in range(len(cascade)):
        print("#"+ "="*60 + "#")
        logging.info("Model performance after stage {:d}".format(stage_id + 1))
        stage_model = cascade[stage_id]
        if opt.cuda:
            stage_model = stage_model.cuda()    
        if action_wise:
            evaluate_action_wise(action_eval_list, stage_model, stats, opt)    
            # update the current estimates and regression targets
            for dataset in action_eval_list:
                dataset.stage_update(stage_model, stats, opt)
        else:
            # evaluate up to this stage
            loss, distance = evaluate(eval_dataset, 
                                      stage_model, 
                                      stats, 
                                      opt, 
                                      save=save, 
                                      save_path=save_path,
                                      procrustes=False, 
                                      per_joint=True, 
                                      apply_dropout=apply_dropout
                                      )

            # update datasets
            eval_dataset.stage_update(stage_model, stats, opt)        
        # release memory
        del stage_model   
    return loss, distance

def logger_print(epoch, 
                 batch_idx, 
                 batch_size, 
                 total_sample, 
                 total_batches,
                 loss
                 ):         
    """
    Log training history.
    """      
    msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
            epoch, 
            batch_idx * batch_size, 
            total_sample,
            100. * batch_idx / total_batches, 
            loss.data.item())
    logging.info(msg)
    return

def train(train_dataset, 
          eval_dataset, 
          model, 
          optim, 
          sche, 
          stats, 
          action_eval_list, 
          opt, 
          plot_loss=False):
    """
    Train a single deep learner.
    """
    x_data = []
    y_data = []
    eval_loss, eval_distance = evaluate(eval_dataset, model, stats, opt)
    if plot_loss:
        import matplotlib.pyplot as plt
        # plot loss curve during training
        ax = plt.subplot(111)
        lines = ax.plot(x_data, y_data)
        plt.xlabel('batch')
        plt.ylabel('training loss')
    for epoch in range(1, opt.epochs + 1):
        model.train()
        # update the learning rate according to the scheduler
        sche.step()        
        # data loader
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=opt.batch_size, 
                                                   shuffle=True, 
                                                   num_workers=opt.num_threads
                                                   )    
        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            data = batch[0]
            target = batch[1]
            if opt.cuda:
                with torch.no_grad():
                    # move to GPU
                    data, target = data.cuda(), target.cuda()                    
            # erase all computed gradient        
            optim.zero_grad()            
            # forward pass to get prediction
            prediction = model(data)
            # compute loss
            loss = F.mse_loss(prediction, target)            
            # smoothed l1 loss function
            #loss = F.smooth_l1_loss(prediction, target)            
            # compute gradient in the computational graph
            loss.backward()            
            # update parameters in the model 
            optim.step()            
            # logging
            if batch_idx % opt.report_every == 0:
                logger_print(epoch,
                             batch_idx,
                             opt.batch_size,
                             len(train_dataset),
                             len(train_loader),
                             loss)                
                x_data.append(num_batches*(epoch-1) + batch_idx)
                y_data.append(loss.data.item())
                if plot_loss:
                    lines[0].set_xdata(x_data)
                    lines[0].set_ydata(y_data)
                    ax.relim()
                    # update ax.viewLim using the new dataLim
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.05)
            # optinal evaluation
            if opt.eval and batch_idx!= 0 and batch_idx % opt.eval_every == 0:
                eval_loss, eval_distance = evaluate(eval_dataset, model, stats, opt)

                # update learning rate if needed
                #sche.step(eval_loss)
                # reset to training mode
                model.train()
        # evaluate after each epoch 
        if opt.eval_action_wise and epoch % 50 == 0:
            evaluate_action_wise(action_eval_list, model, stats, opt)
    logging.info('Training finished.')
    return {'model':model, 'batch_idx':x_data, 'loss':y_data}  

def evaluate_action_wise(dataset_list, model, stats, opt):
    """
    Evaluate for a list of dataset objects, where each contains inputs for one action.
    """
    record_P1 = {}
    record_P2 = {}
    average_P1 = 0
    average_P2 = 0
    protocols = opt.protocols
    for dataset in dataset_list:
        action = dataset.action_name
        if 'P1' in protocols:            
            eval_loss, eval_distance = evaluate(dataset, model, stats, opt, 
                                                verbose=False, procrustes = False)
            record_P1[action] = (eval_loss, eval_distance)
            average_P1 += eval_distance
        if 'P2' in protocols:
            eval_loss, eval_distance = evaluate(dataset, model, stats, opt, 
                                                verbose=False, procrustes = True)
            record_P2[action] = (eval_loss, eval_distance)
            average_P2 += eval_distance            
    average_P1 /= len(dataset_list)
    average_P2 /= len(dataset_list)
    # logging
    for protocol in protocols:
        logging.info("MPJPE under protocol {:s}".format(protocol))
        record = record_P1 if protocol == 'P1' else record_P2
        average = average_P1 if protocol == 'P1' else average_P2
        for key in record.keys():
            logging.info("Action: {:s}, error: {:.2f}".format(key, record[key][1]))
        logging.info("Average error over actions: {:.2f}".format(average))
    return [record_P1, record_P2]

def align_skeleton(skeletons_pred, skeletons_gt, num_of_joints):
    """
    Apply per-frame procrustes alignment before computing MPJPE.
    """
    for j in range(len(skeletons_gt)):
        gt  = np.reshape(skeletons_gt[j,:], [-1,3])
        out = np.reshape(skeletons_pred[j,:],[-1,3])
        _, Z, T, b, c = compute_similarity_transform(gt, 
                                                     out, 
                                                     compute_optimal_scale=True
                                                     )
        out = (b * out.dot(T)) + c
        skeletons_pred[j,:] = np.reshape(out,[-1, (num_of_joints - 1) * 3])     
    return skeletons_pred

def evaluate(eval_dataset, 
             model, 
             stats, 
             opt, 
             save = False, 
             save_path=None,
             verbose = True, 
             procrustes = False, 
             per_joint = False, 
             apply_dropout=False
             ):
    """
    Evaluate a 2D-to-3D lifting model on a given PyTorch dataset.
    Adapted from ICCV 2017 baseline
    https://github.com/una-dinosauria/3d-pose-baseline    
    """
    num_of_joints = 14 if opt.pred14 else 17 
    all_dists = []
    model.eval()
    if apply_dropout:
        def apply_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()        
        # enable the dropout layers to produce a loss similar to the training 
        # loss (only for debugging purpose)
        model.apply(apply_dropout)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, 
                                              batch_size = opt.batch_size, 
                                              shuffle = False, 
                                              num_workers = opt.num_threads
                                              )
    total_loss = 0
    for batch_idx, batch in enumerate(eval_loader):
        data = batch[0]
        target = batch[1]
        if opt.cuda:
            with torch.no_grad():
                data, target = data.cuda(), target.cuda()
        # forward pass to get prediction
        prediction = model(data)
        # mean squared loss 
        loss = F.mse_loss(prediction, target, reduction='sum')
        total_loss += loss.data.item()
        # unnormalize the data
        skeleton_3d_gt = data_utils.unNormalizeData(target.data.cpu().numpy(),
                                                    stats['mean_3d'], 
                                                    stats['std_3d'], 
                                                    stats['dim_ignore_3d']
                                                    )    
        skeleton_3d_pred = data_utils.unNormalizeData(prediction.data.cpu().numpy(),
                                                      stats['mean_3d'], 
                                                      stats['std_3d'], 
                                                      stats['dim_ignore_3d']
                                                      )
        # pick the joints that are used
        dim_use = stats['dim_use_3d']
        skeleton_3d_gt_use = skeleton_3d_gt[:, dim_use]
        skeleton_3d_pred_use = skeleton_3d_pred[:, dim_use]
        # error after a regid alignment, corresponding to protocol #2 in the paper
        if procrustes:
            skeleton_3d_pred_use = align_skeleton(skeleton_3d_pred_use, 
                                                  skeleton_3d_gt_use, 
                                                  num_of_joints
                                                  )
        # Compute Euclidean distance error per joint
        sqerr = (skeleton_3d_gt_use - skeleton_3d_pred_use)**2 # Squared error between prediction and expected output
        dists = np.zeros((sqerr.shape[0], num_of_joints)) # Array with L2 error per joint in mm
        dist_idx = 0
        for k in np.arange(0, num_of_joints*3, 3):
          # Sum across X,Y, and Z dimenstions to obtain L2 distance
          dists[:,dist_idx] = np.sqrt(np.sum(sqerr[:, k:k+3], axis=1))
          dist_idx = dist_idx + 1
        all_dists.append(dists)  
    all_dists = np.vstack(all_dists)
    if per_joint:
        # show average error for each joint
        error_per_joint = all_dists.mean(axis = 0)    
        logging.info('Average error for each joint: ')
        print(error_per_joint)
    avg_loss = total_loss/(len(eval_dataset)*16*3)
    if save:
        record = {'error':all_dists}
        np.save(save_path, np.array(record))
    avg_distance = all_dists.mean()
    if verbose:
        logging.info('Evaluation set: average loss: {:.4f} '.format(avg_loss))
        logging.info('Evaluation set: average joint distance: {:.4f} '.format(avg_distance))
    return avg_loss, avg_distance
