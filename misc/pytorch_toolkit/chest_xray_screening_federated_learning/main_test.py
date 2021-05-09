from Utils import data_utils, model_utils, performance_utils

import torch.nn as nn
import numpy as np
import copy, torch, argparse
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def initialize_testing(site, img_pth, split_npz, test_transform ):
    """Constructs test dataloader, edge indeces and edge attributes"""
    
    data_tst=data_utils.construct_dataset(img_pth, split_npz, site, test_transform, tn_vl_idx=2)
    tst_loader=DataLoader(data_tst, 1, shuffle=False, num_workers=1, pin_memory=False)
    
    edge_index, edge_attr= model_utils.compute_adjacency_matrix('confusion_matrix', site, split_npz)
    
    return tst_loader, edge_index, edge_attr
   
def instantiate_architecture(device, ftr_dim, model_name):
    """ Instantiates architectures"""
    
    inp_dim=512
    backbone_model=models.resnet18(pretrained=True)
    backbone_model.fc=nn.Identity()
    
    cnv_lyr=model_utils.First_Conv()
    fc_layers=model_utils.Fully_Connected_Layer(inp_dim, ftr_dim)
    gnn_model=model_utils.GNN_Network(in_chnls=512, base_chnls=1, grwth_rate=1, depth=1, aggr_md='mean', ftr_dim=4)
    
    ######### These models are globally available ########################
    cnv_lyr.to(device)
    fc_layers.to(device)
    backbone_model.to(device)
    gnn_model.to(device)
    
    return cnv_lyr, backbone_model, fc_layers, gnn_model

def lcl_test(tst_loader,cnv_lyr, backbone_model, fc_layers, gnn_model, edge_index, edge_attr, device):
    """Calls inference function and returns metrics"""
    sens_lst, spec_lst, acc_lst, auc_lst=performance_utils.inference_test(cnv_lyr, backbone_model, fc_layers, gnn_model, tst_loader, edge_index, edge_attr, device)
    return sens_lst, spec_lst, acc_lst, auc_lst

def display(sens_lst, spec_lst, auc_lst):
    """Displays quantitative results"""
    bal_acc=(sens_lst+spec_lst)/2
    tmp=''
    for k in range(0,14):
        tmp=tmp+'%.4f' % auc_lst[k]#+'/'+'%.4f' % bal_acc[k]
        if k<13:
            tmp=tmp+'\t'

    print(tmp)

def main(img_pth, split_npz, test_transform, restart_checkpoint, device):
    """The main function that coordinates inference at different sites"""
    ###### Instantiate the CNN-GNN Architecture ##############
    cnv_lyr, backbone_model, fc_layers, gnn_model=instantiate_architecture(device, ftr_dim=512, model_name='resnet')
    

    #####################################################################################
    ############## Initialize Data Loaders #################
    
    tst_loader0, edge_index0, edge_attr0=initialize_testing(0, img_pth, split_npz, test_transform )
    tst_loader1, edge_index1, edge_attr1=initialize_testing(1, img_pth, split_npz, test_transform )
    tst_loader2, edge_index2, edge_attr2=initialize_testing(2, img_pth, split_npz, test_transform )
    tst_loader3, edge_index3, edge_attr3=initialize_testing(3, img_pth, split_npz, test_transform )
    tst_loader4, edge_index4, edge_attr4=initialize_testing(4, img_pth, split_npz, test_transform )
    
    
    # Load previous checkpoint if resuming the  training else comment out
    checkpoint=torch.load('./Models/'+restart_checkpoint)
    glbl_cnv_wt=checkpoint['cnv_lyr_state_dict']
    glbl_backbone_wt=checkpoint['backbone_model_state_dict']
    glbl_fc_wt=checkpoint['fc_layers_state_dict']
    sit0_gnn_wt=checkpoint['sit0_gnn_model']
    sit1_gnn_wt=checkpoint['sit1_gnn_model']
    sit2_gnn_wt=checkpoint['sit2_gnn_model']
    sit3_gnn_wt=checkpoint['sit3_gnn_model']
    sit4_gnn_wt=checkpoint['sit4_gnn_model']
    del checkpoint
    
    
    ######### Site-0
    cnv_lyr.load_state_dict(glbl_cnv_wt)
    backbone_model.load_state_dict(glbl_backbone_wt)
    fc_layers.load_state_dict(glbl_fc_wt)
    gnn_model.load_state_dict(sit0_gnn_wt)
    
    sens_lst0, spec_lst0, acc_lst0,auc_lst0=lcl_test(tst_loader0,cnv_lyr, backbone_model,fc_layers, gnn_model, 
                                                 edge_index0, edge_attr0, device)
    
    ######### Site-1
    cnv_lyr.load_state_dict(glbl_cnv_wt)
    backbone_model.load_state_dict(glbl_backbone_wt)
    fc_layers.load_state_dict(glbl_fc_wt)
    gnn_model.load_state_dict(sit1_gnn_wt)
    
    sens_lst1, spec_lst1, acc_lst1,auc_lst1=lcl_test(tst_loader1,cnv_lyr, backbone_model,fc_layers, gnn_model, 
                                                 edge_index1, edge_attr1, device)
    
    ######### Site-2
    cnv_lyr.load_state_dict(glbl_cnv_wt)
    backbone_model.load_state_dict(glbl_backbone_wt)
    fc_layers.load_state_dict(glbl_fc_wt)
    gnn_model.load_state_dict(sit2_gnn_wt)
    
    sens_lst2, spec_lst2, acc_lst2,auc_lst2=lcl_test(tst_loader2,cnv_lyr, backbone_model,fc_layers, gnn_model, 
                                                 edge_index2, edge_attr2, device)
    
    ######### Site-3
    cnv_lyr.load_state_dict(glbl_cnv_wt)
    backbone_model.load_state_dict(glbl_backbone_wt)
    fc_layers.load_state_dict(glbl_fc_wt)
    gnn_model.load_state_dict(sit3_gnn_wt)
    
    sens_lst3, spec_lst3, acc_lst3,auc_lst3=lcl_test(tst_loader3,cnv_lyr, backbone_model,fc_layers, gnn_model, 
                                                 edge_index3, edge_attr3, device)
    
    ######### Site-4
    cnv_lyr.load_state_dict(glbl_cnv_wt)
    backbone_model.load_state_dict(glbl_backbone_wt)
    fc_layers.load_state_dict(glbl_fc_wt)
    gnn_model.load_state_dict(sit4_gnn_wt)
    
    sens_lst4, spec_lst4, acc_lst4,auc_lst4=lcl_test(tst_loader4,cnv_lyr, backbone_model,fc_layers, gnn_model, 
                                                 edge_index4, edge_attr4, device)
    
    print('#### Site-0 #####')    
    print('\n'+str(np.mean(auc_lst0))+'\n')
    display(sens_lst0, spec_lst0,auc_lst0)
    print('#### Site-1 #####')
    print('\n'+str(np.mean(auc_lst1))+'\n')
    display(sens_lst1, spec_lst1,auc_lst1)
    print('#### Site-2 #####')
    print('\n'+str(np.mean(auc_lst2))+'\n')
    display(sens_lst2, spec_lst2,auc_lst2)
    print('#### Site-3 #####')
    print('\n'+str(np.mean(auc_lst3))+'\n')
    display(sens_lst3, spec_lst3,auc_lst3)
    print('#### Site-4 #####')
    print('\n'+str(np.mean(auc_lst4))+'\n')
    display(sens_lst4, spec_lst4,auc_lst4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='best_weights.pt', type=str, help='Name of the model to be tested')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_transform=transforms.Compose([
                                    transforms.Resize(320),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
    img_pth='./Data/'
    split_npz='./Data/image_lvl_split.npz'
    main(img_pth, split_npz, test_transform, args.modelname, device)
