import numpy as np
import copy, torch

from torch_geometric.data import Data as Data_GNN
from torch_geometric.data import DataLoader as DataLoader_GNN
from sklearn.metrics import roc_auc_score

def compute_performance(pred, gt):
    """Computes the performance metrics : Accuracy, AUC, Sensitivity and Specificity """
    acc_lst=[]
    auc_lst=[]
    sens_lst=[]
    spec_lst=[]
    
    pred_scr=pred.copy()
    pred_cls=pred.copy()
    
    idx0=np.where(pred_cls<0.5)
    idx1=np.where(pred_cls>=0.5)
    pred_cls[idx0]=0
    pred_cls[idx1]=1
    
    
    
    for cls in range(0, pred_scr.shape[1]):    
        tmp_prd_scr=pred_scr[:,cls]
        tmp_prd_cls=pred_cls[:, cls]
        tmp_gt=gt[:, cls]
        
        TP=np.where((tmp_gt==1) & (tmp_prd_cls==1))[0].shape[0]
        TN=np.where((tmp_gt==0) & (tmp_prd_cls==0))[0].shape[0]
        FP=np.where((tmp_gt==0) & (tmp_prd_cls==1))[0].shape[0]
        FN=np.where((tmp_gt==1) & (tmp_prd_cls==0))[0].shape[0]
        
        acc=(TP+TN)/(TP+TN+FP+FN)
        sens=TP/(TP+FN)
        spec=TN/(TN+FP)
        auc=roc_auc_score(tmp_gt, tmp_prd_scr)
        
        sens_lst.append(sens)
        spec_lst.append(spec)
        acc_lst.append(acc)
        auc_lst.append(auc)
        
    sens_lst=np.array(sens_lst)
    spec_lst=np.array(spec_lst)
    acc_lst=np.array(acc_lst)
    auc_lst=np.array(auc_lst)
    
    return sens_lst, spec_lst, acc_lst, auc_lst


def inference_val(cnv_lyr, backbone_model, fc_layers, gnn_model, val_loader, criterion, edge_index, edge_attr, device):
    """ Inference code for model that is trained with validation data"""
    tot_loss=0
    tot_auc=0
    
    gt_lst=[]
    pred_lst=[]
    
    cnv_lyr.eval()
    backbone_model.eval() 
    fc_layers.eval()
    gnn_model.eval()
    
    with torch.no_grad():
        for count, sample in enumerate(val_loader):
            
            img=sample['img']
            gt=sample['gt']
    
            img=img.to(device)
            gt=gt.to(device)
            
            ##############################################################
            
            img_3chnl=cnv_lyr(img)
            gap_ftr=backbone_model(img_3chnl)
            ftr_lst, _=fc_layers(gap_ftr)
            
            ftr_lst=torch.cat(ftr_lst, dim=1)
    
            data_lst=[]
            for k in range(0, ftr_lst.shape[0]):
                data_lst.append(Data_GNN(x=ftr_lst[k,:,:], edge_index=edge_index, edge_attr=edge_attr, y=torch.unsqueeze(gt[k,:], dim=1))) 
            
            loader = DataLoader_GNN(data_lst, batch_size=ftr_lst.shape[0])
            loader=next(iter(loader)).to(device)
            gt=loader.y
            prd_final=gnn_model(loader)    
    
            if criterion is not None:
                loss=criterion(prd_final, gt)
                tot_loss=tot_loss+loss.cpu().numpy()
                del loss
            
            # Apply the sigmoid
            prd_final=torch.sigmoid(prd_final)
            
            gt_lst.append(gt.cpu().numpy())
            pred_lst.append(prd_final.cpu().numpy())
            
            del gt, prd_final
            
    
    gt_lst=np.concatenate(gt_lst, axis=1)
    pred_lst=np.concatenate(pred_lst, axis=1)
    
    gt_lst=np.transpose(gt_lst)
    pred_lst=np.transpose(pred_lst)
    
    # Now compute and display the average
    count=count+1 # since it began from 0
    if criterion is not None:
        avg_loss=tot_loss/count
    
    sens_lst, spec_lst, acc_lst, auc_lst=compute_performance(pred_lst, gt_lst)
    avg_auc=np.mean(auc_lst)
    
    if criterion is not None:
        print ("\tVal_Loss:  {:.4f},  Avg. AUC: {:.4f}".format(avg_loss, avg_auc))
    else:
        print("\tAvg. AUC: {:.4f}".format(avg_auc))
    
    metric=avg_auc # this will be monitored for Early Stopping
    
    cnv_lyr.train()
    backbone_model.train() 
    fc_layers.train()
    gnn_model.train()
    
    return metric

def inference_test(cnv_lyr, backbone_model, fc_layers, gnn_model, tst_loader,edge_index, edge_attr, device):
    """ Inference code for model that is trained with validation data"""
    tot_auc=0
    
    gt_lst=[]
    pred_lst=[]
    
    cnv_lyr.eval()
    backbone_model.eval() 
    fc_layers.eval()
    gnn_model.eval()
    
    with torch.no_grad():
        for count, sample in enumerate(tst_loader):
            
            img=sample['img']
            gt=sample['gt']
    
            img=img.to(device)
            gt=gt.to(device)
            
            ##############################################################
            
            img_3chnl=cnv_lyr(img)
            gap_ftr=backbone_model(img_3chnl)
            ftr_lst, _=fc_layers(gap_ftr)
            
            ftr_lst=torch.cat(ftr_lst, dim=1)
    
            data_lst=[]
            for k in range(0, ftr_lst.shape[0]):
                data_lst.append(Data_GNN(x=ftr_lst[k,:,:], edge_index=edge_index, edge_attr=edge_attr, y=torch.unsqueeze(gt[k,:], dim=1))) 
            
            loader = DataLoader_GNN(data_lst, batch_size=ftr_lst.shape[0])
            loader=next(iter(loader)).to(device)
            gt=loader.y
            prd_final=gnn_model(loader)    
    
            
            # Apply the sigmoid
            prd_final=torch.sigmoid(prd_final)
            
            gt_lst.append(gt.cpu().numpy())
            pred_lst.append(prd_final.cpu().numpy())
            del gt, prd_final
            
    
    gt_lst=np.concatenate(gt_lst, axis=1)
    pred_lst=np.concatenate(pred_lst, axis=1)
    
    gt_lst=np.transpose(gt_lst)
    pred_lst=np.transpose(pred_lst)
    
    sens_lst, spec_lst, acc_lst, auc_lst=compute_performance(pred_lst, gt_lst)
    avg_auc=np.mean(auc_lst)
    
    cnv_lyr.train()
    backbone_model.train()
    fc_layers.train()
    gnn_model.train()
    
    return sens_lst, spec_lst, acc_lst, auc_lst
