import pickle
import os
import glob
import time
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from SmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp

import torch.multiprocessing as mp
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def get_iou(high_iou, med_iou, low_iou):

class IoUCalculator(object):
    """Computes and stores the current IoU and intersection and union sums"""
    def __init__(self, density):
        self.density = density
        self.IoU = 0
        self.reset()

    def reset(self):
        self.intersection = 0
        self.union = 0

    def update(self, pred, truth):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        curr_int = (pred + truth == 2).sum()
        curr_union = (pred + truth >= 1).sum()
        if curr_union > 0:
            curr_IoU = curr_int / curr_union
        self.intersection += curr_int
        self.union += curr_union

    def all_reduce(self, device):
        total = torch.tensor([self.intersection, self.union], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.intersection, self.union = total.tolist()
        self.IoU = self.intersection / self.union
        return self.intersection, self.union

    def summary(self):
        print("IoU for {} density smoke: {}".format(self.density, self.IoU))



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)



def compute_iou(pred, true, level, iou_dict):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5) * 1
    intersection = (pred + true == 2).sum()
    union = (pred + true >= 1).sum()
    try:
        iou = intersection / union
        iou_dict[level]['int'] += intersection
        iou_dict[level]['union'] += union
        #print('{} density smoke gives: {} IoU'.format(level, iou))
        return iou_dict
    except Exception as e:
        print(e)
        print('there was no {} density smoke in this batch'.format(level))
        return iou_dict

def display_iou(iou_dict):
    high_iou = iou_dict['high']['int']/iou_dict['high']['union']
    med_iou = iou_dict['medium']['int']/iou_dict['medium']['union']
    low_iou = iou_dict['low']['int']/iou_dict['low']['union']
    iou = (iou_dict['high']['int'] + iou_dict['medium']['int'] + iou_dict['low']['int'])/(iou_dict['high']['union'] + iou_dict['medium']['union'] + iou_dict['low']['union'])
    print('OVERALL HIGH DENSITY SMOKE GIVES: {} IoU'.format(high_iou))
    print('OVERALL MEDIUM DENSITY SMOKE GIVES: {} IoU'.format(med_iou))
    print('OVERALL LOW DENSITY SMOKE GIVES: {} IoU'.format(low_iou))
    print('OVERALL OVER ALL DENSITY GIVES: {} IoU'.format(iou))

def val_model(dataloader, model, criterion):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    high_iou = IoUCalculator('high')
    med_iou = IoUCalculator('medium')
    low_iou = IoUCalculator('low')
    iou_dict= {'high': {'int': 0, 'union':0}, 'medium': {'int': 0, 'union':0}, 'low': {'int': 0, 'union':0}}
    device = torch.device(f"cuda:{dist.get_rank()}")
    for data in dataloader:
        batch_data, batch_labels = data
        #batch_data, batch_labels = batch_data.to(device, dtype=torch.float32), batch_labels.to(device, dtype=torch.float32)
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float32, non_blocking=True), batch_labels.to(device, dtype=torch.float32, non_blocking=True)
        preds = model(batch_data)

        high_loss = criterion(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
        med_loss = criterion(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
        low_loss = criterion(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
        loss = 3*high_loss + 2*med_loss + low_loss
        #loss = high_loss + med_loss + low_loss
        test_loss = loss.item()
        total_loss += test_loss
        high_iou.update(preds[:,0,:,:], batch_labels[:,0,:,:])
        med_iou.update(preds[:,1,:,:], batch_labels[:,1,:,:])
        low_iou.update(preds[:,2,:,:], batch_labels[:,2,:,:])
        iou_dict = compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
        iou_dict = compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
        iou_dict = compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
    display_iou(iou_dict)
    final_loss = total_loss/len(dataloader)
    print("Validation Loss: {}".format(round(final_loss,8)), flush=True)
    return final_loss, high_iou, med_iou, low_iou

def train_model(train_dataloader, model, optimizer, criterion):
    device = torch.device(f"cuda:{dist.get_rank()}")
    total_loss = 0.0
    b_sz = len(next(iter(train_dataloader))[0])
    print(f"Batchsize: {b_sz} | Steps: {len(train_dataloader)}")
    model.train()
    torch.set_grad_enabled(True)
    for data in train_dataloader:
        start = time.time()

        optimizer.zero_grad() # zero the parameter gradients
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float32, non_blocking=True), batch_labels.to(device, dtype=torch.float32, non_blocking=True)

        preds = model(batch_data)
        high_loss = criterion(preds[:,0,:,:], batch_labels[:,0,:,:])
        med_loss = criterion(preds[:,1,:,:], batch_labels[:,1,:,:])
        low_loss = criterion(preds[:,2,:,:], batch_labels[:,2,:,:])
        loss = 3*high_loss + 2*med_loss + low_loss
        total_loss += loss.item()

        # compute gradient and do step
        loss.backward()
        optimizer.step()

        print(time.time() - start)
    epoch_loss = total_loss/len(train_dataloader)
    print("Training Loss:   {}".format(round(epoch_loss,8)), flush=True)


def prepare_dataloader(rank, world_size, data_dict, cat, batch_size, pin_memory=True, num_workers=8):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = SmokeDataset(data_dict[cat], data_transforms)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

def main(rank, world_size, exp_num):

    with open('configs/exp{}.json'.format(exp_num)) as fn:
        hyperparams = json.load(fn)

    with open('./dataset_pointers/make_list/subsample.pkl', 'rb') as handle:
        data_dict = pickle.load(handle)

    setup(rank, world_size)
    train_loader = prepare_dataloader(rank, world_size, data_dict, 'train', batch_size=int(hyperparams['batch_size']))
    val_loader = prepare_dataloader(rank, world_size, data_dict, 'val', batch_size=int(hyperparams['batch_size']))

    n_epochs = 100
    start_epoch = 0
    arch = hyperparams['architecture']
    lr = hyperparams['lr']
    best_loss = 10000.0

    model = smp.create_model( # create any model architecture just with parameters, without using its class
            arch=arch,
            encoder_name=hyperparams['encoder'],
            encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3, # model input channels
            classes=3, # model output channels
    )
    model = model.to(rank)
    #model = DDP(model, device_ids=[rank], output_device=rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)


    ckpt_loc = './models/'
    for epoch in range(start_epoch, n_epochs):
        print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        train_model(train_loader, model, criterion, optimizer)
        val_loss, high_iou, med_iou, low_iou = val_model(val_loader, model, criterion)

        if rank==0:
            get_iou(high_iou, med_iou, low_iou)

            best_loss = val_loss
            checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss
                    }
            ckpt_pth = '{}{}_exp{}_{}.pth'.format(ckpt_loc, arch, exp_num, int(time.time()))
            torch.save(checkpoint, ckpt_pth)
            print('SAVING MODEL:\n', ckpt_pth, flush=True)

    dist.destroy_process_group()



def dont_use():
    use_ckpt = False
#use_ckpt = True

    ckpt_list = glob.glob('{}{}_exp{}_*.pth'.format(ckpt_loc, arch, exp_num))
    if use_ckpt:
        ckpt_list.sort()
        if ckpt_list:
            # sorted by time
            most_recent = ckpt_list.pop()
        else:
            use_ckpt = False

    if use_ckpt == True:
        most_recent = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/scripts/deep_learning/models/DLV3P_exp1_1719683871.pth'
        print('using this checkpoint: ', most_recent)
        checkpoint=torch.load(most_recent)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']


if __name__ == '__main__':
# suppose we have 3 gpus
    world_size = 8
    if len(sys.argv) < 2:
        print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
    exp_num = str(sys.argv[1])
    mp.spawn(main, args=(world_size, exp_num), nprocs=world_size, join=True)

