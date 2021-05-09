import numpy as np
import copy, torch


def average_weights(w, cmb_wt, device):
    """Computes weighted average of model weights"""
    cmb_wt=np.array(cmb_wt)
    cmb_wt=cmb_wt.astype(np.float)
    cmb_wt=cmb_wt/np.sum(cmb_wt)

    wts = torch.tensor(cmb_wt).to(device)
    wts=wts.float()
    w_avg = copy.deepcopy(w[0])
    
    for key in w_avg.keys(): # for each layer
        layer = key.split('.')[-1]
    
        if layer == 'num_batches_tracked':
            for i in range(1,len(w)): # for each model
                w_avg[key] += w[i][key].to(device)
            w_avg[key] = torch.div(w_avg[key].float(), torch.tensor(len(w)).float()).to(torch.int64)
        else:
            w_avg[key]=torch.mul(w_avg[key].to(device), wts[0].to(float))
            for i in range(1,len(w)):
                w_avg[key] += torch.mul(w[i][key].to(device), wts[i].to(float))
            
    return w_avg


def aggregate_local_weights(wt0, wt1, wt2, wt3, wt4, device):
    """ Calls 'average_weights'"""
    wt=average_weights([wt0, wt1, wt2, wt3, wt4], 
                       [1.0, 2030.0/1997, 2093.0/1997, 1978.0/1997, 2122.0/1997], device)
    return wt
