from Utils import performance_utils

import copy, torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data as Data_GNN
from torch_geometric.data import DataLoader as DataLoader_GNN

class Custom_Loss(nn.Module):
    """Creates the customized loss function for multilabel classification"""
    def __init__(self, site, device):
        super(Custom_Loss, self).__init__()
        if site==-999:
            wts_pos = np.array([ 6.07201409, 12.57545272,  5.07639982,  1.29352719, 14.83679525,  
                                2.61834939, 9.25154963, 22.75312856, 4.12082252,  7.02592567,  
                                1.58836049, 38.86513797, 15.04438092,  1.17096019])

            wts_neg = np.array([0.68019345, 0.64294624, 0.69547317, 1.16038896, 0.63797481, 0.79812282,
                                0.6549775,  0.62857107, 0.71829276, 0.67000328, 0.99474773, 0.62145383,
                                0.63759652, 1.2806393 ])
        elif site==0:
            wts_pos=np.array([636.94267516,  13.93728223, 5.28038864, 5.26537489, 87.87346221, 8.61623298,
                    67.56756757, 228.83295195, 20.40816327,  79.42811755, 5.70450656, 276.24309392,
                    87.79631255,   5.6840789 ])

            wts_neg=np.array([3.14594016, 4.0373047,  7.68875903, 7.72081532, 3.24612089, 4.91690432, 
                    3.28256303, 3.17389786, 3.69767786, 3.2589213,  6.93769946, 3.16636059, 
                    3.24622626, 6.96815553])
            
        elif site==1:
            wts_pos=np.array([ 31.82686187, 649.35064935, 568.18181818,  11.06439478,  75.75757576, 
                    16.73920321,  11.19319454,  27.94076558,  25.4517689,  158.73015873, 
                    11.25999324, 387.59689922,  88.73114463,   7.74653343])

            wts_neg= np.array([3.40901343, 3.09386795, 3.09597523, 4.26657565, 3.20965464, 3.77330013,
                     4.24772747, 3.46056684, 3.50299506, 3.14011179, 4.23818606, 3.10385499, 
                     3.18989441, 5.11064547])
        elif site==2:
            wts_pos= np.array([653.59477124, 662.25165563, 584.79532164, 4.56350112, 45.12635379, 11.55401502, 
                     675.67567568, 746.26865672, 14.69723692, 29.20560748, 5.70418116, 159.23566879,
                     87.03220191,   5.50721445])

            wts_neg= np.array([3.00057011, 3.00039005, 3.0021916, 8.645284, 3.19856704, 4.02819738, 3.00012, 
                     2.99886043, 3.74868796, 3.3271227,  6.26998558, 3.04395471, 3.09300671, 6.52656311])
        elif site==3:
            wts_pos=np.array([359.71223022, 675.67567568, 515.46391753,   6.02772755,  65.40222368, 16.94053871, 
                    800.0, 740.74074074,19.9960008, 11.29433025, 10.53962901, 78.49293564, 87.2600349, 
                    7.8486775 ])

            wts_neg=np.array([3.18775901, 3.17460317, 3.17924588, 6.64098818, 3.32016335, 3.88424937, 3.1722869, 
                    3.17329356, 3.75276767, 4.38711942, 4.51263538, 3.29228946, 3.27847354, 5.28904638])
        elif site==4:
            wts_pos=np.array([7.84990973, 308.64197531, 454.54545455, 9.28074246, 186.21973929, 16.51800463, 
                    819.67213115, 909.09090909, 27.52546105, 1515.15151515, 10.49538203, 1960.78431373, 
                    47.93863854, 4.16684029])

            wts_neg=np.array([ 4.71720364, 2.97495091, 2.96577496, 4.31723007, 2.99392234, 3.58628604, 
                    2.95718003,  2.95613102, 3.29978551,  2.95229098,  4.09668169,  2.95098415, 
                    3.13952028, 10.06137438])

            
        wts_pos = torch.from_numpy(wts_pos)
        wts_pos = wts_pos.type(torch.Tensor)
        wts_pos=wts_pos.to(device) # size 1 by cls
        
        wts_neg = torch.from_numpy(wts_neg)
        wts_neg = wts_neg.type(torch.Tensor)
        wts_neg=wts_neg.to(device) # size 1 by cls
        
        
        self.wts_pos=wts_pos
        self.wts_neg=wts_neg
        
        #self.bce=nn.BCELoss(reduction='none')
        self.bce=nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, ypred, ytrue):
        
        msk = ((1-ytrue)*self.wts_neg) + (ytrue*self.wts_pos) #1 if ytrue is 0
        #print(msk.shape)
        
        loss=self.bce(ypred,ytrue) # bsz, cls
        loss=loss*msk
        
        loss=loss.view(-1) # flatten all batches and class losses
        loss=torch.mean(loss) 
        
        return loss

def train_end_to_end(cnv_lyr, backbone_model, fc_layers, gnn_model, train_loader, trn_typ, n_batches, criterion, 
                     edge_index, edge_attr, device):
    """Trains model for one local epoch"""
    
    cnv_lyr.train() 
    backbone_model.train()
    fc_layers.train()
    gnn_model.train()
    
    ########## Optimizers and Schedulers #############
    total_batches=len(train_loader)
    lr=10**(-5)
    
    # optimizer
    optim1 = torch.optim.Adam(cnv_lyr.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optim2 = torch.optim.Adam(backbone_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optim3 = torch.optim.Adam(fc_layers.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optim4 = torch.optim.Adam(gnn_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    
    cnt=0
    trn_run_loss=0
    for i, sample in enumerate(train_loader):
        cnt=cnt+1
            
        cnv_lyr, backbone_model, fc_layers, gnn_model, loss,  optim1, optim2, optim3, optim4=train_one_batch(
                        sample, cnv_lyr, backbone_model, fc_layers, gnn_model, optim1, optim2, optim3, optim4, 
                        trn_typ, criterion, edge_index, edge_attr, device)
            
        trn_run_loss=trn_run_loss+loss
            
        if (i+1) % 20== 0: # displays after every 20 batch updates
            print ("cnt {}, Train Loss: {:.4f}".format(cnt,(trn_run_loss/(cnt))), end ="\r")
                
        ############# Monitor Validation Acc and Early Stopping ############
        if cnt>=n_batches:
            break
    return cnv_lyr, backbone_model, fc_layers, gnn_model

def train_one_batch(sample, cnv_lyr, backbone_model, fc_layers, gnn_model, optim1, optim2, optim3, optim4, 
                    trn_typ, criterion, edge_index, edge_attr, device):
    """Trains for one batch update"""
    img=sample['img']
    gt=sample['gt']
    
    img=img.to(device)
    gt=gt.to(device)
    
    ########Forward Pass ##############
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
    
    loss=criterion(prd_final, gt)
    
    ####### Backward Pass ##########
    ### Remove previous gradients
    optim1.zero_grad()
    optim2.zero_grad()
    optim3.zero_grad()
    optim4.zero_grad()
    
    
    ### Compute Gradients
    loss.backward()
    
    ### Optimizer Gradients
    #update weights through backprop using Adam 
    if trn_typ=='full':
        optim1.step()     
        optim2.step()
    
    optim3.step()
    optim4.step()
    
    
    return cnv_lyr, backbone_model, fc_layers, gnn_model, loss,  optim1, optim2, optim3, optim4


def lcl_train(trn_loader, val_loader, criterion, cnv_lyr, backbone_model, fc_layers, gnn_model, edge_index, edge_attr, device):
    """Trains local model and returns validation AuROC, updated model weights"""
    n_batches=1500
    ####### Freeze and train the part which is specific to each site only
    print('Freeze global CNN, fine-tune GNN ...')
    cnv_lyr, backbone_model, fc_layers, gnn_model=train_end_to_end(cnv_lyr, backbone_model, 
                                                fc_layers, gnn_model, trn_loader,'gnn', n_batches, criterion, 
                                                edge_index, edge_attr, device)
    
    ###### Compute the Validation accuracy #######
    print('Computing Validation Performance ...')
    prev_val=performance_utils.inference_val(cnv_lyr, backbone_model, fc_layers, gnn_model, val_loader, criterion,
                        edge_index, edge_attr, device)
    
    ######## Train the entire network in an end-to-end manner ###
    print('Train end-to-end for Local Site ...')
    cnv_lyr, backbone_model, fc_layers, gnn_model=train_end_to_end(cnv_lyr, backbone_model, fc_layers, 
                                                        gnn_model, trn_loader,'full', 2*n_batches, criterion,
                                                        edge_index, edge_attr, device)
    
    
    cnv_wt=copy.deepcopy(cnv_lyr.state_dict())
    backbone_wt=copy.deepcopy(backbone_model.state_dict())
    fc_wt=copy.deepcopy(fc_layers.state_dict())
    gnn_wt=copy.deepcopy(gnn_model.state_dict())
    
    return prev_val, cnv_wt,backbone_wt, fc_wt, gnn_wt
