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
print(f'data_dict: {data_dict_name}')
with open(data_dict_name, 'rb') as handle:
    data_dict = pickle.load(handle)

data_transforms = transforms.Compose([transforms.ToTensor()])

#train_set = SmokeDataset(data_dict['train'], data_transforms)
#val_set = SmokeDataset(data_dict['val'], data_transforms)
test_set = SmokeDataset(data_dict['test'], data_transforms)

#print('there are {} training samples in this dataset'.format(len(train_set)))
print('there are {} testing samples in this dataset'.format(len(test_set)))

def save_test_results(truth_fn, preds, dir_num, iou_dict, category='', iou_type=''):
    save_loc = os.path.join(os.getcwd(),'test_results' , category, str(iou_type), str(dir_num)) # /{}/{}/{}/'.format(category, iou_type, dir_num))
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    print('Saving fn_info and pred to {}'.format(save_loc))

    data_fn = truth_fn.replace('truth', 'data')
    # coords_fn = truth_fn.replace('truth', 'coords')
    skimage.io.imsave(os.path.join(save_loc,'preds.tif'), preds)
    [high_iou, med_iou, low_iou, overall_iou] = get_prev_iou_by_density(iou_dict)

    fn_info = {'data_fn': data_fn,
               'truth_fn': truth_fn,
               # 'coords_fn': coords_fn,
               'low_iou': str(low_iou.cpu().numpy()),
               'medium_iou': str(med_iou.cpu().numpy()),
               'high_iou': str(high_iou.cpu().numpy()),
               'overall_iou': str(overall_iou.cpu().numpy())
               }
    json_object = json.dumps(fn_info, indent=4)
    
    with open(os.path.join(save_loc,"fn_info.json"), "w") as outfile:
        outfile.write(json_object)

def test_model(dataloader, model, BCE_loss):
    best_iou_dict = {'low':{'iou': [], 'idx': []}, 'medium': {'iou': [], 'idx': []}, 'high': {'iou': [], 'idx': []}, 'overall': {'iou': [], 'idx': []}}
    worst_iou_dict = {'low': {'iou': [], 'idx': []}, 'medium':{'iou': [], 'idx': []}, 'high': {'iou': [], 'idx': []}, 'overall': {'iou': [], 'idx': []}}
    model.eval()
    torch.set_grad_enabled(False)
    top_n=20
    iou_dict= {'high': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'medium': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'low': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}}

    print('Calculating IoUs for testing data...')
    for idx, data in enumerate(dataloader):
        # if idx < 3: ##### RM
        batch_data, batch_labels, truth_fn = data
        # print('1', batch_data.shape, batch_labels.shape) ### #RM
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)
        iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
        iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
        iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
        best_iou_dict, worst_iou_dict = save_iou(iou_dict, best_iou_dict, worst_iou_dict, idx, top_n)

    # print(best_iou_dict, worst_iou_dict)
    print('Sorting best/worst IoUs...')
    # Sort the lists to only grab the top/bottom N 
    best_iou_dict = sort_and_clip_iou_dict(best_iou_dict, top_n, best=True)
    worst_iou_dict = sort_and_clip_iou_dict(worst_iou_dict, top_n, best=False)
    print(best_iou_dict, worst_iou_dict)
    # index the data loader 
    for density in best_iou_dict:
        for idx in best_iou_dict[density]['idx']:
            print('Saving best IoU cases...')
            # index the data loader,
            # for i, _ in enumerate(dataloader):
            batch_data, batch_labels, truth_fn = dataloader.dataset[idx]
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            # print('2', batch_data.shape, batch_labels.shape) #### RM
            batch_labels = torch.unsqueeze(batch_labels, 0); batch_data = torch.unsqueeze(batch_data, 0); 
            # print('3', batch_data.shape, batch_labels.shape) #### RM
            preds = model(batch_data )
            iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
            iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
            iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
            save_test_results(truth_fn, preds.detach().to('cpu').numpy(), idx, iou_dict, category=density, iou_type='best')
        for idx in worst_iou_dict[density]['idx']:
            print('Saving worst IoU cases...')
            # print('2', batch_data.shape, batch_labels.shape) #### RM
            batch_data, batch_labels, truth_fn = dataloader.dataset[idx]
            # print('2', batch_data.shape, batch_labels.shape) #### RM
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            # print('2', batch_data.shape, batch_labels.shape) #### RM
            batch_labels = torch.unsqueeze(batch_labels, 0); batch_data = torch.unsqueeze(batch_data, 0); 
            # print('3', batch_data.shape, batch_labels.shape) #### RM
            preds = model(batch_data )
            iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
            iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
            iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
            save_test_results(truth_fn, preds.detach().to('cpu').numpy(), idx, iou_dict, category=density, iou_type='worst')
    return 

def sort_and_clip_iou_dict(iou_dict, top_n, best=True):
    for density in iou_dict:
        # Zip IoU and idx together to sort them
        paired_iou_idx = list(zip(iou_dict[density]['iou'], iou_dict[density]['idx']))

        # Sort by IoU values, in revFerse order for best IoUs, normal for worst IoUs
        paired_iou_idx.sort(key=lambda x: x[0], reverse=best)

        # Clip to the top N entries
        clipped_iou_idx = paired_iou_idx[:top_n]
        new_iou=[]; new_idx=[]
        for iou, idx in clipped_iou_idx:
            new_iou.append(iou); new_idx.append(idx)
        iou_dict[density]['iou'] = new_iou
        iou_dict[density]['idx'] = new_idx

    return iou_dict

def save_iou(iou_dict, best_iou_dict, worst_iou_dict, idx, best_threshold=0.6, worst_threshold=0.2): 
    [high_iou, med_iou, low_iou, overall_iou] = get_prev_iou_by_density(iou_dict)
    if high_iou > best_threshold:
        best_iou_dict['high']['iou'].append(high_iou)
        best_iou_dict['high']['idx'].append(idx)
    if med_iou > best_threshold:
        best_iou_dict['medium']['iou'].append(med_iou)
        best_iou_dict['medium']['idx'].append(idx)
    if low_iou > best_threshold:
        best_iou_dict['low']['iou'].append(low_iou)
        best_iou_dict['low']['idx'].append(idx)
    if overall_iou > best_threshold:
        best_iou_dict['overall']['iou'].append(overall_iou)
        best_iou_dict['overall']['idx'].append(idx)
    if high_iou < worst_threshold:
        worst_iou_dict['high']['iou'].append(high_iou)
        worst_iou_dict['high']['idx'].append(idx)
    if med_iou < worst_threshold:
        worst_iou_dict['medium']['iou'].append(med_iou)
        worst_iou_dict['medium']['idx'].append(idx)
    if low_iou < worst_threshold:
        worst_iou_dict['low']['iou'].append(low_iou)
        worst_iou_dict['low']['idx'].append(idx)
    if overall_iou < worst_threshold:
        worst_iou_dict['overall']['iou'].append(overall_iou)
        worst_iou_dict['overall']['idx'].append(idx)

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

