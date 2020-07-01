import logging

import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np

def normalize(vec):
    # normalize a numpy vector
    return (vec-vec.mean())/vec.std()

def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):
    """
    Un-normalizes a matrix whose mean has been substracted and that has been divided by
    standard deviation. Some dimensions might also be missing
    
    Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
    Returns
    orig_data: the input normalized_data, but unnormalized
    """
    T = normalized_data.shape[0] # Batch size
    D = data_mean.shape[0] # Dimensionality
    
    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = np.array([dim for dim in range(D)
                                    if dim not in dimensions_to_ignore])
    
    orig_data[:, dimensions_to_use] = normalized_data
    
    # Multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, array_2d, array_3d, split, action_name=None, refine_3d=False):
        """
        Args:
            
        """
        self.data_2d = array_2d
        self.data_3d = array_3d
        self.num_samples = len(self.data_2d)
        self.split = split
        self.action_name = action_name
        self.refine_3d = refine_3d
        self.stage_idx = 1
        # initialize current estimate 3d pose
        self.current_estimate = np.zeros(self.data_3d.shape, dtype=np.float32)
        # initialize the regression target (starts with zero estimate)
        self.regression_target = self.data_3d.copy()
        assert len(self.data_2d) == len(self.data_3d)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.refine_3d and self.stage_idx > 1:
            #return np.concatenate([self.data_2d[idx], self.current_estimate[idx]]), self.regression_target[idx]
            # normalized version
            return np.concatenate([self.data_2d[idx], normalize(self.current_estimate[idx])]), self.regression_target[idx]
        else:
            return self.data_2d[idx], self.regression_target[idx]
    
    def set_stage(self, stage_idx):
        self.stage_idx = stage_idx
        return
    
    def stage_update(self, model, stats, opt, verbose=False):
        # update the dataset for cascaded regression
        model.eval()
        eval_loader = torch.utils.data.DataLoader(self, 
                                                  batch_size = opt.batch_size, 
                                                  shuffle = False, 
                                                  num_workers = opt.num_threads) 
        # vector to add at last
        update_vector = []
        total_loss = 0
        all_distance = np.zeros((0))
        for batch_idx, batch in enumerate(eval_loader):
            data = batch[0]
            target = batch[1]
            if opt.cuda:
                with torch.no_grad():
                    # move to GPU
                    data, target = data.cuda(), target.cuda()
            # forward pass to get prediction
            prediction = model(data)
            # mean squared loss 
            loss = F.mse_loss(prediction, target, reduction='sum')
            total_loss += loss.data.item()
            # compute distance of body joints in un-normalized format
            unnorm_target = unNormalizeData(target.data.cpu().numpy(), 
            stats['mean_3d'], stats['std_3d'], stats['dim_ignore_3d']) 
            # put the prediction into the update list
            prediction = prediction.data.cpu().numpy()
            update_vector.append(prediction)
            unnorm_pred = unNormalizeData(prediction, 
            stats['mean_3d'], stats['std_3d'], stats['dim_ignore_3d'])
            # pick the joints that are used
            dim_use = stats['dim_use_3d']
            unnorm_target_use = unnorm_target[:, dim_use]
            unnorm_target_use = unnorm_target_use.reshape(-1,16,3)
            unnorm_pred_use = unnorm_pred[:, dim_use]
            unnorm_pred_use = unnorm_pred_use.reshape(-1,16,3)
            distance = np.sum((unnorm_target_use - unnorm_pred_use)**2, axis=2)
            distance = np.mean(np.sqrt(distance), axis=1)
            all_distance = np.hstack([all_distance, distance])      
        # update the current estimate and regression target
        update_vector = np.concatenate(update_vector, axis=0)
        self.current_estimate += update_vector
        self.regression_target -= update_vector
        # report statistics
        avg_loss = total_loss/(self.num_samples*16*3)
        avg_distance = all_distance.mean()
        if verbose:
            logging.info('Stage update finished.')
            logging.info('{:s} set: average loss: {:.4f} '.format(self.split, avg_loss))
            logging.info('{:s} set: average joint distance: {:.4f} '.format(self.split, avg_distance))
        return avg_loss, avg_distance