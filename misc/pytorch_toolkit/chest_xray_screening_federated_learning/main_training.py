from Utils import data_utils, model_utils, train_utils, average_utils

import torch.nn as nn
import copy, torch, os
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def initialize_training(site, img_pth, split_npz, train_transform, test_transform, device):
    """Constructs dataloader, loss function, edge indeces and edge attributes"""
    
    b_sz=8
    
    data_trn=data_utils.construct_dataset(img_pth, split_npz, site, train_transform, tn_vl_idx=0)
    trn_loader=DataLoader(data_trn, b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    
    data_val=data_utils.construct_dataset(img_pth, split_npz, site, test_transform, tn_vl_idx=1)
    val_loader=DataLoader(data_val, 1, shuffle=False, num_workers=1, pin_memory=False)
    
    criterion=train_utils.Custom_Loss(site, device)
    
    edge_index, edge_attr= model_utils.compute_adjacency_matrix('confusion_matrix', site, split_npz)
    
    
    return trn_loader, val_loader, criterion, edge_index, edge_attr



def initialize_model_weights(cnv_lyr, backbone_model, fc_layers, gnn_model):
    """Creates copy of model state dicts"""
    
    cnv_wt=copy.deepcopy(cnv_lyr.state_dict())
    backbone_wt=copy.deepcopy(backbone_model.state_dict())
    fc_wt=copy.deepcopy(fc_layers.state_dict())
    gnn_wt=copy.deepcopy(gnn_model.state_dict())
    
    return cnv_wt, backbone_wt, fc_wt, gnn_wt




def save_model_weights(glbl_cnv_wt, glbl_backbone_wt, glbl_fc_wt, sit0_gnn_wt, sit1_gnn_wt, 
                       sit2_gnn_wt,sit3_gnn_wt, sit4_gnn_wt, mx_nm):
    """Saves models: Both global and local"""
    torch.save({
                'cnv_lyr_state_dict': glbl_cnv_wt,
                'backbone_model_state_dict': glbl_backbone_wt,
                'fc_layers_state_dict': glbl_fc_wt,
                'sit0_gnn_model': sit0_gnn_wt,
                'sit1_gnn_model': sit1_gnn_wt,
                'sit2_gnn_model': sit2_gnn_wt,
                'sit3_gnn_model': sit3_gnn_wt,
                'sit4_gnn_model': sit4_gnn_wt,
                }, './Models/'+mx_nm)
    
    
def instantiate_architecture(device, ftr_dim,):
    """ Instantiates architectures"""
  
    inp_dim=512
    backbone_model=models.resnet18(pretrained=True)
    backbone_model.fc=nn.Identity()
    
    cnv_lyr=model_utils.First_Conv()
    fc_layers=model_utils.Fully_Connected_Layer(inp_dim, ftr_dim)
    gnn_model=model_utils.GNN_Network(in_chnls=512, base_chnls=1, grwth_rate=1, depth=1, aggr_md='mean', ftr_dim=4)
    
    cnv_lyr.to(device)
    fc_layers.to(device)
    backbone_model.to(device)
    gnn_model.to(device)
    
    return cnv_lyr, backbone_model, fc_layers, gnn_model

def main(img_pth, split_npz, train_transform, test_transform, device, restart_checkpoint=''):
    """The main function that coordinates training at different sites"""
    ###### Instantiate the CNN-GNN Architecture ##############
    cnv_lyr, backbone_model, fc_layers, gnn_model=instantiate_architecture(device, ftr_dim=512)
    

    #####################################################################################
    ############## Initialize Data Loaders #################
    split_npz='./Data/image_lvl_split.npz'
    
    trn_loader0, val_loader0, criterion0, edge_index0, edge_attr0=initialize_training(0, img_pth, split_npz, 
                                                              train_transform, test_transform, device)
    
    trn_loader1, val_loader1, criterion1, edge_index1, edge_attr1=initialize_training(1, img_pth, split_npz, 
                                                              train_transform, test_transform, device)
    
    trn_loader2, val_loader2, criterion2, edge_index2, edge_attr2=initialize_training(2, img_pth, split_npz, 
                                                              train_transform, test_transform, device)
    
    trn_loader3, val_loader3, criterion3, edge_index3, edge_attr3=initialize_training(3, img_pth, split_npz, 
                                                              train_transform, test_transform, device)
    
    trn_loader4, val_loader4, criterion4, edge_index4, edge_attr4=initialize_training(4, img_pth, split_npz, 
                                                              train_transform, test_transform, device)
    
    #########################################################################################
    ### Initialize local and global model weights with the Imagenet pre-trained weights for backbone 
    #and identical model weights for the other layers.

    
    glbl_cnv_wt, glbl_backbone_wt, glbl_fc_wt, gnn_wt=initialize_model_weights(cnv_lyr, backbone_model, 
                                                                                    fc_layers, gnn_model)
    sit0_gnn_wt=copy.deepcopy(gnn_wt)
    sit1_gnn_wt=copy.deepcopy(gnn_wt)
    sit2_gnn_wt=copy.deepcopy(gnn_wt)
    sit3_gnn_wt=copy.deepcopy(gnn_wt)
    sit4_gnn_wt=copy.deepcopy(gnn_wt)
    
    del gnn_wt
    
    # Load previous checkpoint if resuming the  training else comment out
    if (restart_checkpoint!=''):
        checkpoint=torch.load(restart_checkpoint)
        glbl_cnv_wt=checkpoint['cnv_lyr_state_dict']
        glbl_backbone_wt=checkpoint['backbone_model_state_dict']
        glbl_fc_wt=checkpoint['fc_layers_state_dict']
        sit0_gnn_wt=checkpoint['sit0_gnn_model']
        sit1_gnn_wt=checkpoint['sit1_gnn_model']
        sit2_gnn_wt=checkpoint['sit2_gnn_model']
        sit3_gnn_wt=checkpoint['sit3_gnn_model']
        sit4_gnn_wt=checkpoint['sit4_gnn_model']
    
    ##########################################################################################
    ################ Begin Actual Training ############
    max_epochs=2
    max_val=0
    for epoch in range(0, max_epochs):
        print('############ Epoch: '+str(epoch)+'   #################')
        
        ###### Load the global model weights ########
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit0_gnn_wt)
        
        print('\n \n SITE 0 \n')
        prv_val0, sit0_cnv_wt,sit0_backbone_wt, sit0_fc_wt, sit0_gnn_wt=train_utils.lcl_train(trn_loader0, val_loader0,
                                                criterion0, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index0, edge_attr0, device)
        
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit1_gnn_wt)
        
        print('\n \n SITE 1 \n')
        prv_val1, sit1_cnv_wt,sit1_backbone_wt, sit1_fc_wt, sit1_gnn_wt=train_utils.lcl_train(trn_loader1, val_loader1, 
                                                criterion1, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index1, edge_attr1, device)
        
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit2_gnn_wt)
        
        print('\n \n SITE 2 \n')
        prv_val2, sit2_cnv_wt,sit2_backbone_wt, sit2_fc_wt, sit2_gnn_wt=train_utils.lcl_train(trn_loader2, val_loader2, 
                                                criterion2, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index2, edge_attr2, device)
        
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit3_gnn_wt)
        
        print('\n \n SITE 3 \n')
        prv_val3, sit3_cnv_wt,sit3_backbone_wt, sit3_fc_wt, sit3_gnn_wt=train_utils.lcl_train(trn_loader3, val_loader3, 
                                                criterion3, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index3, edge_attr3, device)
        
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit4_gnn_wt)
        
        print('\n \n SITE 4 \n')
        prv_val4, sit4_cnv_wt,sit4_backbone_wt, sit4_fc_wt, sit4_gnn_wt=train_utils.lcl_train(trn_loader4, val_loader4, 
                                                criterion4, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index4, edge_attr4, device)

        
        avg_val=(prv_val0+prv_val1+prv_val2+prv_val3+prv_val4)/5
        print('Avg Val AUC: '+str(avg_val))
        
        if avg_val>max_val:
            max_val=avg_val
            save_model_weights(glbl_cnv_wt, glbl_backbone_wt, glbl_fc_wt, sit0_gnn_wt, sit1_gnn_wt, sit2_gnn_wt, sit3_gnn_wt, sit4_gnn_wt, 'best_weights.pt')
            print('Validation Performance Improved !')
            
            
        ############### Compute the global model weights #############
        
        glbl_cnv_wt=average_utils.aggregate_local_weights(sit0_cnv_wt, sit1_cnv_wt, sit2_cnv_wt,
                                                sit3_cnv_wt, sit4_cnv_wt, device)
        
        glbl_backbone_wt=average_utils.aggregate_local_weights(sit0_backbone_wt, sit1_backbone_wt, sit2_backbone_wt,
                                                sit3_backbone_wt, sit4_backbone_wt, device)
        
        glbl_fc_wt=average_utils.aggregate_local_weights(sit0_fc_wt, sit1_fc_wt, sit2_fc_wt, sit3_fc_wt, sit4_fc_wt, device)


if __name__ == '__main__':
    
    img_pth='./Data/'
    split_npz='./Data/image_lvl_split.npz'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isdir('./Models'):
        os.makedirs('./Models')
    ################# Data Augmentation and Transforms #####################
    # Training Transformations/ Data Augmentation
    train_transform=transforms.Compose([
                                    transforms.Resize(350),
                                    transforms.RandomResizedCrop(320, scale=(0.8, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
    
    # Test/Val Transformations
    test_transform=transforms.Compose([
                                    transforms.Resize(320),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
    
    main(img_pth, split_npz, train_transform, test_transform, device)
    
