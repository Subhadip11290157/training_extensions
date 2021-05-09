import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import NNConv
from torch_geometric.nn import BatchNorm as GNN_BatchNorm
from sklearn.metrics import cohen_kappa_score


###################### Define the CNN Architectures ########################
# Define Architecture
''' Instead of creating an instance of the models within the constructor, 
 We will pass the initial layer for 1 to 3 channel, the backbone model 
 and the FC layers part separately as input arguments. 
 
 This will allow us to simply load the weights for each CNN model separately, 
 may be useful when updating only part of the network
 
 '''
class Fully_Connected_Layer(nn.Module):
    """Creates instance of the fully connected layers"""
    def __init__(self, inp_dim, ftr_dim):
        super(Fully_Connected_Layer, self).__init__()
        
        ftr_lyr=nn.ModuleList()
        cls_lyr=nn.ModuleList()
        
        for cls in range(0,14):
            ftr_lyr.append(
                            nn.Sequential(
                                            nn.Linear(inp_dim, ftr_dim, bias=False), 
                                            nn.BatchNorm1d(ftr_dim),
                                            nn.ReLU(),
                                            nn.Linear(ftr_dim, ftr_dim, bias=False), 
                                            nn.BatchNorm1d(ftr_dim),    
                                            nn.ReLU()
                                        )
                          )
            
            cls_lyr.append(
                            nn.Sequential(
                                            nn.Linear(ftr_dim, 1, bias=False), 
                                            nn.BatchNorm1d(1)
                                        )
                            )
            self.ftr_lyr=ftr_lyr
            self.cls_lyr=cls_lyr

    def forward(self, x):
        prd_lst=[]
        ftr_lst=[]
        for cls in range(0,14):
            
            ftr=self.ftr_lyr[cls](x)
            ftr_lst.append(torch.unsqueeze(ftr, dim=1))
            prd=self.cls_lyr[cls](ftr)
            prd_lst.append(prd)
            
        
        prd=torch.cat(prd_lst, axis=1)
        return ftr_lst, prd

class First_Conv(nn.Module):
    """Creates instance of first convolutional layer"""
    def __init__(self):
        super(First_Conv, self).__init__()
        
        # Convert 1 channel to 3 channel also can be made unique for each site
        self.convert_channels=nn.Sequential(nn.Conv2d(1,3,1,1, bias=False), 
                                            nn.BatchNorm2d(3),
                                            nn.ReLU() )
    def forward(self, x):
        x=self.convert_channels(x)
        return x

def compute_edge_attr(A):
    """Uses a dense Graph where all neighbors are considered"""
    edge=[]
    edge_attr=[]
    for j in range(0,14):
        for k in range(0,14):
            if j==k:
                continue
            edge.append(np.array([j,k]))
            edge_attr.append(A[j,k])
    
    edge=np.array(edge)
    edge_attr=np.array(edge_attr)
    
    edge=torch.from_numpy(np.transpose(edge))
    edge=edge.long()
    
    edge_attr=torch.from_numpy(edge_attr)
    edge_attr=torch.unsqueeze(edge_attr, dim=1)
    edge_attr=edge_attr.float()
       
    return edge, edge_attr


def compute_adjacency_matrix(adj_type, site, split_npz='image_lvl_split.npz'):
    """Computes adjacency matrix"""
    # load the npz file
    a=np.load(split_npz, allow_pickle=True)    
    gt=a['gt']
    clstr_assgn=a['clstr_assgn']
    trn_val_tst=a['trn_val_tst']
    del a
    
    if site==-999:
        idx=np.where(trn_val_tst==0)[0]
    else:
        idx=np.where((clstr_assgn==site) & (trn_val_tst==0))[0]
    
    gt=gt[idx,:]
    
    kappa=np.zeros((14,14))
    TP=np.zeros((14,14))
    TN=np.zeros((14,14))
    FP=np.zeros((14,14))
    FN=np.zeros((14,14))
    kappa=np.zeros((14,14))
    agree=np.zeros((14,14))
    P1=np.zeros((14,14))
    
    
    for j in range(0,14):
        gt_j=gt[:,j]
        for k in range(0, 14):
            gt_k=gt[:,k]
            
            ## Kappa and agree are symmetric ie., A(i,j)=A(j,i)
            kappa[j,k]=cohen_kappa_score(gt_j, gt_k)
            agree[j,k]=(np.where(gt_j==gt_k)[0].shape[0])/gt.shape[0]
            
            # How many times are both j and k =1---> This will be symmetric
            TP[j,k]=(np.where((gt_j==1) & (gt_k==1))[0].shape[0])/gt.shape[0]
            # How many times are both j and k=0 ---> This will be symmetric
            TN[j,k]=(np.where((gt_j==0) & (gt_k==0))[0].shape[0])/gt.shape[0]
            
            ####### FP and FN will get reversed for A(i,j) and A(j,i)
            # How many time k is 1 but j is 0
            FP[j,k]=(np.where((gt_j==0) & (gt_k==1))[0].shape[0])/gt.shape[0]
            # How many time k is 0 but j is 1
            FN[j,k]=(np.where((gt_j==1) & (gt_k==0))[0].shape[0])/gt.shape[0]
            
    if adj_type=='kappa':
        A=kappa
    elif adj_type=='fraction_agreement':
        A=agree
    elif adj_type=='confusion_matrix':
        A=np.concatenate((np.expand_dims(TP, axis=2), np.expand_dims(TN, axis=2),np.expand_dims(FP, axis=2), np.expand_dims(FN, axis=2)), axis=2)
                    
    if A.ndim==2:
        tmp_edge, edge_attr=compute_edge_attr(A)
    else:
        edge_lst=[]
        edge_attr_lst=[]
        for x in range(A.shape[2]):
            tmp_edge, tmp_edge_attr=compute_edge_attr(np.squeeze(A[:,:,x]))
            edge_lst.append(tmp_edge)
            edge_attr_lst.append(tmp_edge_attr)
            
            
        edge_attr=torch.cat(edge_attr_lst, dim=1)
      
    return tmp_edge, edge_attr


################## GNN Architecture Classes #############

class create_mlp(nn.Module):
    """
    Creates instance of MLP that maps the edge weight to the weights used to avg.
    The features from the neighbors"""
    def __init__(self, in_chnl, out):
        super(create_mlp, self).__init__() 
        
        self.lyr=nn.Sequential(
                                nn.Linear(in_chnl, out, bias=True),
                                #nn.BatchNorm1d(out),
                                nn.Tanh()
                                    )
        
    def forward(self, x):
        out=self.lyr(x)
        return out

#######################################################################################

class Res_Graph_Conv_Lyr(nn.Module):
    """Creates instance for Resdiual Block for the GNN"""
    def __init__(self, in_chnls, base_chnls, mlp_model, aggr_md):
        super(Res_Graph_Conv_Lyr, self).__init__() 
        
        self.GNN_lyr=NNConv(in_chnls, base_chnls, mlp_model, aggr=aggr_md)
        self.bn=GNN_BatchNorm(base_chnls)
        
    def forward(self, x, edge_index, edge_attr):
        h=self.GNN_lyr(x, edge_index, edge_attr)
        h=self.bn(h)
        h=F.relu(h)
        
        return (x+h)
        #return h
    
########################################################################################

class GNN_Network(nn.Module):
    """Creates instance of the Graph Convolution Network"""
    def __init__(self, in_chnls, base_chnls, grwth_rate, depth, aggr_md, ftr_dim):
        super(GNN_Network, self).__init__()
        
        my_gcn=nn.ModuleList()
        
        # Base channels is actually the fraction of inp.
        in_chnls=int(in_chnls)
        base_chnls=int(base_chnls*in_chnls)
        
        # A GCN to map input channels to base channels dimensions
        my_gcn.append(Res_Graph_Conv_Lyr(in_chnls, base_chnls, create_mlp(ftr_dim, in_chnls*base_chnls), aggr_md))
        
        in_chnls=base_chnls
        for k in range(0, depth):
            out_chnls=int(in_chnls*grwth_rate)
            # Get a GCN
            in_chnls=max(in_chnls,1)
            out_chnls=max(out_chnls,1)
            my_gcn.append(Res_Graph_Conv_Lyr(in_chnls, out_chnls, create_mlp(ftr_dim,in_chnls*out_chnls), aggr_md))
            in_chnls=out_chnls
        
        #### Add the final classification layer that will convert output to 1D 
        my_gcn.append(NNConv(in_chnls, 1, create_mlp(ftr_dim, 1*in_chnls), aggr='mean'))
        
        self.my_gcn=my_gcn
        self.dpth=depth
        
    
    def forward(self, data):
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        cnt=0
        x=self.my_gcn[cnt](x, edge_index, edge_attr)
        #x=F.relu(x)
        
        #print('entering loop ...')
        for k in range(0, self.dpth):
            cnt=cnt+1
            #print(cnt)
            x=self.my_gcn[cnt](x, edge_index, edge_attr)
            #x=F.relu(x)
        
        #print('out of loop...')
        cnt=cnt+1
        out=self.my_gcn[cnt](x, edge_index, edge_attr) # num_nodes by 1
        #out=F.sigmoid(out)
        return out
