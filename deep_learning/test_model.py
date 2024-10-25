import pickle
import json
import skimage
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from TestSmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
from metrics import compute_iou, display_iou, get_prev_iou_by_density

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if len(sys.argv) < 2:
    print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
if len(sys.argv) > 2:
    print("IN TEST MODE!")
    test_mode = sys.argv[2]
else:
    test_mode = False


exp_num = str(sys.argv[1])
data_dict_name = str(sys.argv[3])
#data_dict_name = 'pseudo_labeled_ds.pkl'
#data_dict_name = '/projects/mecr8410/SmokeViz_code/deep_learning/dataset_pointers/midday/less_than_3hrs.pkl'
#region = 'NE'
#data_dict_name = '/projects/mecr8410/SmokeViz_code/deep_learning/dataset_pointers/geo_dependent/{}.pkl'.format(region)
print(data_dict_name)
with open(data_dict_name, 'rb') as handle:
    data_dict = pickle.load(handle)
#print(data_dict)

data_transforms = transforms.Compose([transforms.ToTensor()])

#train_set = SmokeDataset(data_dict['train'], data_transforms)
#val_set = SmokeDataset(data_dict['val'], data_transforms)
test_set = SmokeDataset(data_dict['test'], data_transforms)

#print('there are {} training samples in this dataset'.format(len(train_set)))
print('there are {} testing samples in this dataset'.format(len(test_set)))

def save_test_results(truth_fn, preds, dir_num, iou_dict):
    save_loc = '/projects/mecr8410/smoke/plot_results/test_results/{}/'.format(dir_num)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    truth_fn = truth_fn[0]
    data_fn = truth_fn.replace('truth', 'data')
    coords_fn = truth_fn.replace('truth', 'coords')
    skimage.io.imsave(save_loc + 'preds.tif', preds)
    low_iou = iou_dict['low']['prev_int']/iou_dict['low']['prev_union']
    medium_iou = iou_dict['medium']['prev_int']/iou_dict['medium']['prev_union']
    high_iou = iou_dict['high']['prev_int']/iou_dict['high']['prev_union']
    overall_int = iou_dict['low']['prev_int'] + iou_dict['medium']['prev_int'] + iou_dict['high']['prev_int']
    overall_union = iou_dict['low']['prev_union'] + iou_dict['medium']['prev_union'] + iou_dict['high']['prev_union']
    overall_iou = overall_int / overall_union

    fn_info = {'data_fn': data_fn,
               'truth_fn': truth_fn,
               'coords_fn': coords_fn,
               'low_iou': str(low_iou.cpu().numpy()),
               'medium_iou': str(medium_iou.cpu().numpy()),
               'high_iou': str(high_iou.cpu().numpy()),
               'overall_iou': str(overall_iou.cpu().numpy())
               }
    json_object = json.dumps(fn_info, indent=4)
    with open(save_loc + "fn_info.json", "w") as outfile:
        outfile.write(json_object)

def test_model(dataloader, model, BCE_loss):
    best_iou_dict = {'low':{'iou': [], 'idx': []}, 'medium': {'iou': [], 'idx': []}, 'high': {'iou': [], 'idx': []}, 'overall': {'iou': [], 'idx': []}}
    worst_iou_dict = {'low': {'iou': [], 'idx': []}, 'medium':{'iou': [], 'idx': []}, 'high': {'iou': [], 'idx': []}, 'overall': {'iou': [], 'idx': []}}

    model.eval()
    torch.set_grad_enabled(False)
    # total_loss = 0.0
    iou_dict= {'high': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'medium': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'low': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}}
    #max_num = 100
    for idx, data in enumerate(dataloader):
        batch_data, batch_labels, truth_fn = data
        print(len(batch_data))
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)

        # high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
        # med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
        # low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
        # loss = 3*high_loss + 2*med_loss + low_loss
        # #loss = high_loss + med_loss + low_loss
        # test_loss = loss.item()
        # total_loss += test_loss
        iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
        iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
        iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
        best_iou_dict, worst_iou_dict = save_ious(iou_dict)

    # Sort the lists to only grab the top/bottom N  
    # index the data loader 
    for idx in best_iou_dict[][]:
        for i, _ in enumerate(dataloader):
            batch_data, batch_labels, truth_fn = data
            if i is in best_iou_dict[][]:
                # make and save model pred.
        
    #    if idx < max_num:
    #        print(idx)
    #        save_test_results(truth_fn, preds.detach().to('cpu').numpy(), idx, iou_dict)
        #    break
    # display_iou(iou_dict)
    # final_loss = total_loss/len(dataloader)
    # print("Testing Loss: {}".format(round(final_loss,8)), flush=True)
    return _ # final_loss

def save_ious(iou_dict, best_iou_dict, worst_iou_dict, idx):
    [high_iou, med_iou, low_iou, overall_iou] =get_prev_iou_by_density(iou_dict)
    if high_iou > 0.95:
        best_iou_dict['high']['iou'].append(high_iou)
        best_iou_dict['high']['idx'].append(idx)
    if med_iou >0.95:
        best_iou_dict['medium']['iou'].append(med_iou)
        best_iou_dict['medium']['idx'].append(idx) 
    if low_iou >0.95:
        best_iou_dict['low']['iou'].append(low_iou)
        best_iou_dict['low']['idx'].append(idx) 
    if overall_iou >0.95:
        best_iou_dict['overall']['iou'].append(overall_iou)
        best_iou_dict['overall']['idx'].append(idx) 

    #do the worst

    return best_iou_dict, worst_iou_dict
def val_model(dataloader, model, BCE_loss):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0

    iou_dict= {'high': {'int': 0, 'union':0}, 'medium': {'int': 0, 'union':0}, 'low': {'int': 0, 'union':0}}
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)

        high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
        med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
        low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
        loss = 3*high_loss + 2*med_loss + low_loss
        #loss = high_loss + med_loss + low_loss
        test_loss = loss.item()
        total_loss += test_loss
        iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
        iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
        iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
    display_iou(iou_dict)
    final_loss = total_loss/len(dataloader)
    print("Validation Loss: {}".format(round(final_loss,8)), flush=True)
    return final_loss

def train_model(train_dataloader, val_dataloader, model, n_epochs, start_epoch, exp_num, BCE_loss):
    history = dict(train=[], val=[])
    best_loss = 10000.0

    for epoch in range(start_epoch, n_epochs):
        total_loss = 0.0
        print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)
        model.train()
        torch.set_grad_enabled(True)
        #for batch_data, batch_labels in train_dataloader:
        for data in train_dataloader:
            batch_data, batch_labels = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            #print(torch.isnan(batch_data).any())
            optimizer.zero_grad() # zero the parameter gradients
            preds = model(batch_data)
            high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
            med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
            low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
            loss = 3*high_loss + 2*med_loss + low_loss
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            total_loss += train_loss
        epoch_loss = total_loss/len(train_dataloader)
        print("Training Loss:   {0}".format(round(epoch_loss,8), epoch+1), flush=True)
        val_loss = val_model(val_dataloader, model, BCE_loss)
        history['val'].append(val_loss)
        history['train'].append(epoch_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }
            torch.save(checkpoint, '/scratch/alpine/mecr8410/semantic_segmentation_smoke/scripts/deep_learning/models/checkpoint_exp{}.pth'.format(exp_num))
            #torch.save(model, './scratch/alpine/mecr8410/semantic_segmentation_smoke/scripts/deep_learning/models/best_model.pth')
    print(history)
    return model, history


with open('configs/exp{}.json'.format(exp_num)) as fn:
    hyperparams = json.load(fn)

use_ckpt = False
#use_ckpt = True
# BATCH_SIZE = int(hyperparams["batch_size"])
BATCH_SIZE = 1
#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

n_epochs = 100
start_epoch = 0
model = smp.DeepLabV3Plus(
#model = smp.Unet(
        #encoder_name="resnext101_32x8d", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_name="timm-efficientnet-b2",
        #encoder_name=hyperparams['encoder'],
        encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3, # model input channels
        classes=3, # model output channels
)
model = model.to(device)
lr = hyperparams['lr']
optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
if use_ckpt == True:
    checkpoint=torch.load('/scratch/alpine/mecr8410/semantic_segmentation_smoke/scripts/deep_learning/models/checkpoint_exp{}.pth'.format(exp_num))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

BCE_loss = nn.BCEWithLogitsLoss()
if test_mode:
    print("IN TEST MODE!")
    #chkpt_pth = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/scripts/deep_learning/models/checkpoint.pth'
    chkpt_pth = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/SmokeViz_code/deep_learning/models/DeepLabV3Plus_exp0_1729717558.pth'
    print(chkpt_pth)
    checkpoint=torch.load(chkpt_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_model(test_loader, model, BCE_loss)
else:
    train_model(train_loader, val_loader, model, n_epochs, start_epoch, exp_num, BCE_loss)

