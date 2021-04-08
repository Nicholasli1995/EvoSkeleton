"""
Utility functions.
"""
import libs.dataset.h36m.data_utils as data_utils
import libs.dataset.h36m.cameras as cameras
import libs.dataset.h36m.pth_dataset as dataset
import libs.visualization.viz as viz

import logging
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
input_size = 32
output_size = 48

# global configurations
camera_frame = True # camera coordinate
predict_14 = False # predict 14 joints

def save_ckpt(opt, record, stats):
    """
    Save training results.
    """
    cascade = record['cascade']
    if not opt.save:
        return False
    if opt.save_name is None:
        save_name = time.asctime()
        save_name += ('stages_' + str(opt.num_stages) + 'blocks_' 
                      + str(opt.num_blocks) + opt.extra_str
                      )        
    save_dir = os.path.join(opt.save_root, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(cascade, os.path.join(save_dir, 'model.th'))
    np.save(os.path.join(save_dir, 'stats.npy'), stats)
    print('Model saved at ' + save_dir)
    return True

def load_ckpt(opt):
    cascade = torch.load(os.path.join(opt.ckpt_dir, 'model.th'))
    stats = np.load(os.path.join(opt.ckpt_dir, 'stats.npy'), allow_pickle=True).item()
    if opt.cuda:
        cascade.cuda()
    return cascade, stats

def list_remove(list_a, list_b):
    list_c = []
    for item in list_a:
        if item not in list_b:
            list_c.append(item)
    return list_c


def get_all_data(data_x, data_y, camera_frame):
    """
    Obtain a list of all the batches, randomly permutted
    Args
      data_x: dictionary with 2d inputs
      data_y: dictionary with 3d expected outputs
      camera_frame: whether the 3d data is in camera coordinates
      training: True if this is a training batch. False otherwise.
    Returns
      encoder_inputs: list of 2d batches
      decoder_outputs: list of 3d batches
    """

    # Figure out how many frames we have
    n = 0
    for key2d in data_x.keys():
      n2d, _ = data_x[ key2d ].shape
      n = n + n2d

    encoder_inputs  = np.zeros((n, input_size), dtype=float)
    decoder_outputs = np.zeros((n, output_size), dtype=float)

    # Put all the data into big arrays
    idx = 0
    for key2d in data_x.keys():
      (subj, b, fname) = key2d
      # keys should be the same if 3d is in camera coordinates
      key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
      key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d

      n2d, _ = data_x[ key2d ].shape
      encoder_inputs[idx:idx+n2d, :]  = data_x[ key2d ]
      decoder_outputs[idx:idx+n2d, :] = data_y[ key3d ]
      idx = idx + n2d

    return encoder_inputs, decoder_outputs

def adjust_figure(left = 0, right = 1, bottom = 0.01, top = 0.95,
                  wspace = 0, hspace = 0.4):  
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    return

def visualize(eval_dataset, model, stats, opt, save=False, save_dir=None):
    # visualze model prediction batch by batch
    batch_size = 9
    # how many batches to save
    if save:
        num_batches = 10
        current_batch = 1
    model.eval()
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size, 
                                              shuffle = True, num_workers = opt.num_threads)     
    for batch_idx, batch in enumerate(eval_loader):
        if save and current_batch > num_batches:
            break
        data = batch[0]
        target = batch[1]
        if opt.cuda:
            with torch.no_grad():
                # move to GPU
                data, target = data.cuda(), target.cuda()
        # forward pass to get prediction
        prediction = model(data)
        # un-normalize the data
        skeleton_2d = data_utils.unNormalizeData(data.data.cpu().numpy(), 
        stats['mean_2d'], stats['std_2d'], stats['dim_ignore_2d'])
        skeleton_3d_gt = data_utils.unNormalizeData(target.data.cpu().numpy(), 
        stats['mean_3d'], stats['std_3d'], stats['dim_ignore_3d'])    
        skeleton_3d_pred = data_utils.unNormalizeData(prediction.data.cpu().numpy(), 
        stats['mean_3d'], stats['std_3d'], stats['dim_ignore_3d'])
        # visualizing
        if save:
            plt.ioff()
        f = plt.figure(figsize=(16, 8))
        axes = []
        for sample_idx in range(batch_size):
            ax = plt.subplot(3, 9, 3*sample_idx + 1)
            viz.show2Dpose(skeleton_2d[sample_idx], ax)
            plt.gca().invert_yaxis()
            ax = plt.subplot(3, 9, 3*sample_idx + 2, projection='3d')
            viz.show3Dpose(skeleton_3d_gt[sample_idx], ax)
            ax = plt.subplot(3, 9, 3*sample_idx + 3, projection='3d')
            viz.show3Dpose(skeleton_3d_pred[sample_idx], ax, pred=True)  
            viz.show3Dpose(skeleton_3d_gt[sample_idx], ax, gt=True)   
            axes.append(ax)      
        adjust_figure(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95,
                      wspace = 0.3, hspace = 0.3)              
        if not save:
            plt.pause(0.5)
            # rotate the axes and update
            for angle in range(0, 360, 5):
                for ax in axes:
                    ax.view_init(30, angle)
                plt.draw()
                plt.pause(.001)
            input('Press enter to view next batch.')
        else:
            # save plot
            f.savefig(save_dir +'/'+ str(current_batch) + '.png')
        plt.close(f)     
        del axes
        if save:
            current_batch += 1
    return

def temp_visualize(eval_dataset, model, stats, opt):
    # visualze model prediction batch by batch
    model.eval()
    data = np.load('./pics/pts1.npy').astype(np.float32)
    data = data[:,2:]
    # normalize the data
    mean_vec = stats['mean_2d'][stats['dim_use_2d']]
    std_vec = stats['std_2d'][stats['dim_use_2d']]
    data = (data-mean_vec)/std_vec
    data = torch.from_numpy(data.astype(np.float32))
    data = data.cuda()
    # forward pass to get prediction
    prediction = model(data)
    # un-normalize the data
    skeleton_2d = data_utils.unNormalizeData(data.data.cpu().numpy(), 
    stats['mean_2d'], stats['std_2d'], stats['dim_ignore_2d'])   
    skeleton_3d_pred = data_utils.unNormalizeData(prediction.data.cpu().numpy(), 
    stats['mean_3d'], stats['std_3d'], stats['dim_ignore_3d'])
    # visualizing
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    viz.show2Dpose(skeleton_2d[0], ax)
    plt.gca().invert_yaxis()
    ax = plt.subplot(1, 2, 2, projection='3d')
    viz.show3Dpose(skeleton_3d_pred[0], ax, pred=True)              
    plt.show()
            # rotate the axes and update
#            for angle in range(0, 360, 5):
#                for ax in axes:
#                    ax.view_init(30, angle)
#                plt.draw()
#                plt.pause(.001)
#            input('Press enter to view next batch.')
    return

def visualize_cascade(eval_dataset, cascade, stats, opt, save=False, save_dir=None):
    num_stages = len(cascade)
    # visualze model prediction batch by batch
    batch_size = 5
    # how many batches to save
    if save:
        num_batches = 10
        current_batch = 1
    for stage_model in cascade:
        stage_model.eval()
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size, 
                                              shuffle = False, 
                                              num_workers = opt.num_threads)     
    for batch_idx, batch in enumerate(eval_loader):
        if save and current_batch > num_batches:
            break
        data = batch[0]
        ## debug
        # enc_in = np.array([[648., 266], [679, 311], [688, 320], [693, 161],
        #            [620, 244], [526, 156], [642, 160], [590, 310],
        #            [505, 350], [380, 375], [491, 285],
        #            [543, 190], [572, 119], [515, 417], [518, 514],
        #            [512, 638]],dtype=np.float32)
        enc_in = data
        enc_in = enc_in.reshape(1, 32)
        # normalize
        data_mean_2d = stats['mean_2d']
        dim_to_use_2d = stats['dim_use_2d']
        data_std_2d = stats['std_2d']
        enc_in = (enc_in - data_mean_2d[dim_to_use_2d])/data_std_2d[dim_to_use_2d]
        data = torch.from_numpy(enc_in.astype(np.float32))
        ## End experiment 2019/10/16
        target = batch[1]
        # store predictions for each stage
        prediction_stages = []
        if opt.cuda:
            with torch.no_grad():
                # move to GPU
                data, target = data.cuda(), target.cuda()
        # forward pass to get prediction for the first stage
        prediction = cascade[0](data)
        prediction_stages.append(prediction)
        # prediction for later stages
        for stage_idx in range(1, num_stages):
            prediction = cascade[stage_idx](data)
            prediction_stages.append(prediction_stages[stage_idx-1] + prediction)
        # un-normalize the data
        skeleton_2d = data_utils.unNormalizeData(data.data.cpu().numpy(), 
        stats['mean_2d'], stats['std_2d'], stats['dim_ignore_2d'])
        skeleton_3d_gt = data_utils.unNormalizeData(target.data.cpu().numpy(), 
        stats['mean_3d'], stats['std_3d'], stats['dim_ignore_3d'])
        for stage_idx in range(num_stages):
            prediction_stages[stage_idx] = data_utils.unNormalizeData(prediction_stages[stage_idx].data.cpu().numpy(), 
            stats['mean_3d'], stats['std_3d'], stats['dim_ignore_3d'])
        ## save intermediate results
        # import scipy.io as sio
        # p3d = prediction_stages[0]
        # sio.savemat('./teaser_pose3d.mat', {'pred_3d':p3d.reshape(32,3),
        #             'pred_2d':np.array([[648., 266], [679, 311], [688, 320], [693, 161],
        #            [620, 244], [526, 156], [642, 160], [590, 310],
        #            [505, 350], [447, 348], [380, 375], [491, 285],
        #            [543, 190], [572, 119], [515, 417], [518, 514],
        #            [512, 638]])})        
        ## End Experiment 2019/10/16
        # visualizing
        if save:
            plt.ioff()
        f = plt.figure(figsize=(16, 8))
        axes = []
        for sample_idx in range(batch_size):
            for stage_idx in range(num_stages):
                ax = plt.subplot(batch_size, num_stages+1, 1+(num_stages+1)*sample_idx)
                viz.show2Dpose(skeleton_2d[sample_idx], ax)
                plt.gca().invert_yaxis()
                ax = plt.subplot(batch_size, num_stages+1, 
                                 2+stage_idx+(num_stages+1)*sample_idx, projection='3d')
                viz.show3Dpose(prediction_stages[stage_idx][sample_idx], ax, pred=True)  
                viz.show3Dpose(skeleton_3d_gt[sample_idx], ax, gt=True)   
                axes.append(ax)      
        adjust_figure(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95,
                      wspace = 0.3, hspace = 0.3)              
        if not save:
            plt.pause(0.5)
            # rotate the axes and update
#            for angle in range(0, 360, 5):
#                for ax in axes:
#                    ax.view_init(30, angle)
#                plt.draw()
#                plt.pause(.001)
            input('Press enter to view next batch.')
        else:
            # save plot
            f.savefig(save_dir +'/'+ str(current_batch) + '.png')
        plt.close(f)     
        del axes
        if save:
            current_batch += 1    
    return

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
      X: array NxM of targets, with N number of points and M point dimensionality
      Y: array NxM of inputs
      compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
      d: squared error after transformation
      Z: transformed Y
      T: computed rotation
      b: scaling
      c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)
    return d, Z, T, b, c
