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
# data_dict_name = 'pseudo_labeled_ds.pkl'
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
    save_loc = os.path.join(os.getcwd(), 'test_results/{}/{}/'.format(ckpt_fn, dir_num))
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
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    iou_dict= {'high': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'medium': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'low': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}}
    #max_num = 100
    for idx, data in enumerate(dataloader):
        batch_data, batch_labels, truth_fn = data
        print(len(batch_data))
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
       # if #idx < max_num:
       #     print(idx)
       #     save_test_results(truth_fn, preds.detach().to('cpu').numpy(), idx, iou_dict)
       #     break
    display_iou(iou_dict)
    final_loss = total_loss/len(dataloader)
    print("Testing Loss: {}".format(round(final_loss,8)), flush=True)
    return final_loss

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


def test_model_track_iou(dataloader, model, BCE_loss):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0

    top_n = 20  # number of best/worst cases to track
    best_cases = {'low': [], 'medium': [], 'high': [], 'overall': []}
    worst_cases = {'low': [], 'medium': [], 'high': [], 'overall': []}

    iou_dict= {'high': {'int': 0, 'union':0}, 'medium': {'int': 0, 'union':0}, 'low': {'int': 0, 'union':0}}
    
    for idx, data in enumerate(dataloader):
        batch_data, batch_labels, truth_fn = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)

        # print('len(preds), len(batch_labels), len(truth_fn)', len(preds), len(batch_labels), len(truth_fn)) ### 
        if idx > 3:
            sys.exit()

        # Go through every sample individually to get individual sample IoU scores rather than IoU across the batch
        for pred, true, curr_truth_fn in zip(preds, batch_labels, truth_fn):
            # print('pred.shape, true.shape, curr_truth_fn', pred.shape, true.shape, curr_truth_fn) ###
            iou_dict= compute_iou(pred[0,:,:], true[0,:,:], 'high', iou_dict)
            iou_dict= compute_iou(pred[1,:,:], true[1,:,:], 'medium', iou_dict)
            iou_dict= compute_iou(pred[2,:,:], true[2,:,:], 'low', iou_dict)
    
            [high_iou, med_iou, low_iou, overall_iou] = get_prev_iou_by_density(iou_dict)
            # print('[high_iou, med_iou, low_iou, overall_iou]', [high_iou, med_iou, low_iou, overall_iou])###

            print(f"IoUs for {curr_truth_fn} - High: {high_iou:.4f}, Medium: {med_iou:.4f}, Low: {low_iou:.4f}, Overall: {overall_iou:.4f}")

            track_iou_cases(best_cases['low'], worst_cases['low'], curr_truth_fn, low_iou, top_n)
            track_iou_cases(best_cases['medium'], worst_cases['medium'], curr_truth_fn, med_iou, top_n)
            track_iou_cases(best_cases['high'], worst_cases['high'], curr_truth_fn, high_iou, top_n)
            track_iou_cases(best_cases['overall'], worst_cases['overall'], curr_truth_fn, overall_iou, top_n)

            # print('best_cases', best_cases)
            # print('worst_cases', worst_cases)

    return best_cases, worst_cases

def track_iou_cases(best_list, worst_list, truth_fn, iou_score, top_n):
    if not torch.isnan(iou_score): # nan iou means the union was 0 and thus no pixels were of that level were in either truth or pred. 
        iou_score = iou_score.cpu().item()
        best_list.append((truth_fn, iou_score))
        worst_list.append((truth_fn, iou_score))

        if len(best_list) > top_n:
            best_list.sort(key=lambda x: x[1], reverse=True)  # Sort best_list in descending order (highest IoU first)
            best_list.pop()  # Remove the lowest IoU score from best_list
        if len(worst_list) > top_n:
            worst_list.sort(key=lambda x: x[1])  # Sort worst_list in ascending order (lowest IoU first)
            worst_list.pop()  # Remove the highest IoU score from worst_list

    print(f"Best List: {[iou for _, iou in best_list]}")
    print(f"Worst List: {[iou for _, iou in worst_list]}")

            
def compute_and_save_tracked_cases(dataloader, model, best_cases, worst_cases):
    model.eval()
    torch.set_grad_enabled(False)

    for idx, data in enumerate(dataloader):
        batch_data, batch_labels, truth_fn = data
        batch_data = batch_data.to(device, dtype=torch.float)
        preds = model(batch_data)
        
        # Loop through batch items
        for pred, true, curr_truth_fn in zip(preds, batch_labels, truth_fn): 
            for iou_type in ['low', 'medium', 'high', 'overall']:
                # Check for the best cases
                for (tracked_truth_fn, iou_score) in best_cases[iou_type]:
                    if curr_truth_fn == tracked_truth_fn:
                        save_test_results(curr_truth_fn, pred.detach().cpu().numpy(), 'best', iou_type, iou_score)

                # Check for the worst cases
                for (tracked_truth_fn, iou_score) in worst_cases[iou_type]:
                    if curr_truth_fn == tracked_truth_fn:
                        save_test_results(curr_truth_fn, pred.detach().cpu().numpy(), 'worst', iou_type, iou_score)



def save_test_results(truth_fp, preds, category, iou_type, iou_score):
    save_loc = os.path.join(os.getcwd(), f'test_results/{category}/{iou_type}/{iou_score:.4f}')
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    skimage.io.imsave(os.path.join(save_loc, 'preds.tif'), preds)

    fn_info = {
        'truth_fp': truth_fp,
        'iou_score': str(iou_score) #.cpu().item()
    }
    with open(os.path.join(save_loc, "fn_info.json"), "w") as outfile:
        json.dump(fn_info, outfile, indent=4)

with open('configs/exp{}.json'.format(exp_num)) as fn:
    hyperparams = json.load(fn)

use_ckpt = False
#use_ckpt = True
BATCH_SIZE = int(hyperparams["batch_size"])
# BATCH_SIZE = 128
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
    chkpt_pth = './models/DeepLabV3Plus_exp1_1726250084.pth' #'./models/DLV3P_exp1_1719683871.pth'
    print(chkpt_pth)
    checkpoint=torch.load(chkpt_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    # test_model(test_loader, model, BCE_loss)
    best_cases, worst_cases = test_model_track_iou(test_loader, model, BCE_loss)
    # compute_and_save_tracked_cases(test_loader, model, best_cases, worst_cases)
else:
    train_model(train_loader, val_loader, model, n_epochs, start_epoch, exp_num, BCE_loss)
