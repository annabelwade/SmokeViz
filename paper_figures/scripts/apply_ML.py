import shutil
import sys
import os
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from SmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
import numpy as np
import time, json

from plot_results import plot_test_results

global full_data_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_data_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/full_dataset/'
Mie_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie2/'

def load_ckpt(exp_num, base_model=False, print_history=False):    
    with open('/scratch1/RDARCH/rda-ghpcs/Annabel.Wade/semantic_segmentation_smoke/scripts/deep_learning/configs/exp{}.json'.format(exp_num)) as fn:
        hyperparams = json.load(fn)
    use_ckpt = hyperparams['use_chkpt']
    encoder_weights = None
    model = smp.create_model(
            arch=hyperparams['architecture'],
            encoder_name=hyperparams['encoder'],
            encoder_weights=encoder_weights,
            in_channels=3,          
            classes=3,
    )
    lr = hyperparams['lr']
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr) 
    # model = nn.DataParallel(model, device_ids=[i for i in range(num_GPUs)])
    model = model.to(device)
    best_val_iou = -100000.0
    ckpt_fp, checkpoint = None, None

    if use_ckpt:
        if base_model:
            # Load base model
            ckpt_fp = '/scratch1/RDARCH/rda-ghpcs/Annabel.Wade/semantic_segmentation_smoke/scripts/deep_learning/models/DeepLabV3Plus_exp1_1726250084.pth'
        else:
            use_best_model =''
            if 'use_best_model' in hyperparams:
                use_best_model = '_best' if (hyperparams['use_best_model'] == "True") else ''
            ckpts_lst = glob.glob('/scratch1/RDARCH/rda-ghpcs/Annabel.Wade/semantic_segmentation_smoke/scripts/deep_learning/models/{}_exp{}{}_*.pth'.format(hyperparams['architecture'], exp_num, use_best_model))
            
            if (len(exp_num)>1) and (len(ckpts_lst) == 0):
                temp_exp_num = exp_num[0]
                ckpts_lst = glob.glob('/scratch1/RDARCH/rda-ghpcs/Annabel.Wade/semantic_segmentation_smoke/scripts/deep_learning/models/{}_exp{}_*.pth'.format(hyperparams['architecture'], temp_exp_num))
            
            if len(ckpts_lst) > 0:
                ckpt_fp = max(ckpts_lst)
            else: 
                raise FileNotFoundError('no ckpt for exp_num {}'.format(exp_num))
            
        checkpoint=torch.load(ckpt_fp, map_location=torch.device(device))#'./models/{}_exp{}_*.pth'.format(hyperparams['architecture'], exp_num)) # insert exp_num of ckpt
        state_dict = checkpoint['model_state_dict']
        first_key = next(iter(state_dict))
        if first_key.startswith('module.'):
            new_state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict  # 'module.' is not present in keys

        # if not first_key.startswith('module.'):
        #     new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        # else:
        #     new_state_dict = state_dict  # 'module.' is already present in keys
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'history' in checkpoint:
            history = checkpoint['history']
            if print_history: print('History loaded from checkpoint:', history)
        else:
            print('No history found in the checkpoint.')

        if 'val_iou' in checkpoint:
            best_val_iou = checkpoint['val_iou']
        print('Loading from checkpoint at {} (epoch {}, best_val_iou {})'.format(ckpt_fp, start_epoch, best_val_iou))
#        best_loss = checkpoint['loss']
        print('ckpt {}\n best_val_iou: {} \nepoch: {}'.format(ckpt_fp, best_val_iou, start_epoch))
    return ckpt_fp, checkpoint, model, optimizer


def get_file_list(yr, dn, idx):
    dn = str(dn).zfill(3)
    truth_file_list = []
    truth_file_list = glob.glob('{}truth/{}/*/{}/*_{}.tif'.format(full_data_dir, yr, dn, idx))
    print('{}truth/{}/*/{}/*_{}.tif'.format(full_data_dir, yr, dn, idx))
    truth_file_list.sort()
    print(truth_file_list)
    data_file_list = [s.replace('truth','data') for s in truth_file_list]
    print('number of samples for idx:', len(truth_file_list))
    data_dict = {'find': {'truth': truth_file_list, 'data': data_file_list}}
    return data_dict

def compute_heavy_iou(preds, truths):
    densities = ['heavy', 'medium', 'low']
    intersection = 0
    union = 0
    for idx, level in enumerate(densities):
        pred = preds[:,idx,:,:]
        true = truths[:,idx,:,:]
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        intersection += (pred + true == 2).sum()
        union += (pred + true >= 1).sum()
        break
    try:
        iou = intersection / union
        return iou
    except Exception as e:
        print(e)
    return 0

def compute_iou(preds, truths):
    densities = ['heavy', 'medium', 'low']
    intersection = 0
    union = 0
    for idx, level in enumerate(densities):
        pred = preds[:,idx,:,:]
        true = truths[:,idx,:,:]
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        intersection += (pred + true == 2).sum()
        union += (pred + true >= 1).sum()
    try:
        iou = intersection / union
        return iou
    except Exception as e:
        print(e)
    return 0


def run_model(yr, dn, idx, chkpt_path = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/SmokeViz_code/pl_derived_ds/models/ckpt2.pth'):

    data_dict = get_file_list(yr,dn,idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transforms = transforms.Compose([transforms.ToTensor()])

    test_set = SmokeDataset(data_dict['find'], data_transforms)

    #print('there are {} images for this annotation'.format(len(test_set)))

    def get_best_file(dataloader, model):
        model.eval()
        torch.set_grad_enabled(False)
        # iou has to be more than .01
        best_iou = .3
        ious = []
        labels = []
        preds = []
        data = []
        best_truth_fn = None
        for idx, data in enumerate(dataloader):
            batch_data, batch_labels, truth_fn = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            out = model(batch_data)
            iou = compute_heavy_iou(out, batch_labels)
            iou = iou.cpu().detach().numpy()
            ious.append(np.round(iou, 5))
            labels.append(batch_labels.squeeze(0).cpu().detach().numpy())
            pred = torch.sigmoid(out)
            pred = (pred > 0.5) * 1
            preds.append(pred.squeeze(0).cpu().detach().numpy())
            data.append(batch_data.squeeze(0).cpu().detach().numpy())

        return ious, preds, labels, data


    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    model = smp.DeepLabV3Plus(
            encoder_name="timm-efficientnet-b2",
            encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3, # model input channels
            classes=3, # model output channels
    )
    model = model.to(device)
    #chkpt_path = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/SmokeViz_code/pl_derived_ds/models/ckpt2.pth'
    checkpoint = torch.load(chkpt_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    ious, preds, labels,data = get_best_file(test_loader, model)
    print(ious)

    return ious, preds, labels, data

def get_data_dict_from_fn(truth_fn):
    data_fn = truth_fn.replace('truth','data')
    data_dict = {'find': {'truth': [truth_fn], 'data': [data_fn]}}
    return data_dict

def get_pred(dataloader, model, device):
    model.eval()
    torch.set_grad_enabled(False)
    ious = []
    for idx, data in enumerate(dataloader):
        batch_data, batch_labels, truth_fn = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        out = model(batch_data)
        iou = compute_iou(out, batch_labels)
        iou = iou.cpu().detach().numpy()
        print(np.round(iou, 4))
        pred = torch.sigmoid(out)
        pred = (pred > 0.5) * 1
        pred = pred.squeeze(0).cpu().detach().numpy()
    return pred, iou

def run_model_single_fn(fn, chkpt_path = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/SmokeViz_code/pl_derived_ds/models/ckpt2.pth', model=''):
    data_dict = get_data_dict_from_fn(fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([transforms.ToTensor()])
    test_set = SmokeDataset(data_dict['find'], data_transforms)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    if model == '':
        model = smp.DeepLabV3Plus(
                encoder_name="timm-efficientnet-b2",
                encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3, # model input channels
                classes=3, # model output channels
        )
        model = model.to(device)
        #chkpt_path = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/SmokeViz_code/pl_derived_ds/models/ckpt2.pth'
        checkpoint = torch.load(chkpt_path, map_location=torch.device(device), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    pred, iou = get_pred(test_loader, model, device)
    return pred, iou
fn = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Light/051/G16_s20220512030203_33.79_-81.5_92.tif'
#pred = run_model_single_fn(fn)
#plot_test_results(pred, fn, save_fig=True)
#run_model(2022, 40, 123)
