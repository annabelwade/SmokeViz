import pickle
from glob import glob
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
from metrics import *
import matplotlib.pyplot as plt
import skimage
from datetime import datetime
from tqdm import tqdm
from testing_ckpt_utils import *
from tabulate import tabulate
from Loss import DiceLoss, CombinedLoss, get_loss_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_GPUs = torch.cuda.device_count()
print(device, num_GPUs)
num_cores = os.cpu_count()
print(f"CPU cores {num_cores}")

if len(sys.argv) < 2:
    print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
params_to_show_list = []

# Add parameter selection for results table
if len(sys.argv) < 3:
    param_to_show = 'architecture'  # default to showing architecture type
    print('\n No parameter specified for results table. Defaulting to architecture type.', flush=True)
else:
    for param in sys.argv[2:]: # add all parameters to show in results table
        params_to_show_list.append(param)
    print(f'\n Showing {params_to_show_list} in results table.', flush=True)
    
# Input format #
# 1_1.2.1_3_T
# => test exp 1, 1.2.1, 3 individually; and ensemble them together
# 1_F
# => test exp 1; don't ensemble
# 0_1.0.2_T
# => test base model and exp 1.0.2 indivdually; and ensemble them togethwer
input = sys.argv[1]
input_list = str(input).split('_')
ensemble = input_list[-1]
input_list.remove(ensemble)
print(input, input_list, ensemble)

dict_fp = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/SmokeViz_code/deep_learning/dataset_pointers/pseudo/pseudo.pkl'
print('Loading dataset...', flush=True)
with open(dict_fp, 'rb') as handle:
    data_dict = pickle.load(handle)

data_transforms = transforms.Compose([transforms.ToTensor()])
test_set = SmokeDataset(data_dict['test'], data_transforms)

print('there are {} testing samples in {}'.format(len(test_set), dict_fp))

BCE_loss = nn.BCEWithLogitsLoss()
BATCH_SIZE = 1
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    drop_last=True,
    num_workers=num_cores,
    pin_memory=True
)

# results_table = [["Experiment",'Epoch', param_to_show, "High IoU",  "Medium IoU", "Low IoU",  "Overall IoU"]]

# # ### Testing individual exp_num
# print('================== INDIVIDUAL MODEL RESULTS ==================')
# for exp_num in tqdm(input_list, desc="Testing models"):
#     if exp_num[0:4] == '1.6.':
#         region = get_param_from_config(exp_num, 'region')
#         dict_fp = f'geo_split/{region}.pkl'
#         with open(dict_fp, 'rb') as handle:
#             data_dict = pickle.load(handle)
#         test_set = SmokeDataset(data_dict['test'], data_transforms)
#         test_loader = torch.utils.data.DataLoader(
#             dataset=test_set, 
#             batch_size=BATCH_SIZE, 
#             shuffle=False, 
#             drop_last=True,
#         )

#     print(f"\nTesting experiment {exp_num}")
#     loss_fn = get_loss_function(exp_num)
#     ckpt_fp, checkpoint, model, optimizer = load_ckpt(exp_num)
#     arch_name = os.path.split(ckpt_fp)[1].split('_')[1]
#     final_loss, iou_dict, iou_list = test_model(test_loader, model, loss_fn, ckpt_fp, exp_num)
#     param_value = get_param_from_config(exp_num, param_to_show)
#     results_table.append([exp_num, checkpoint['epoch'], param_value] + iou_list)

# print(tabulate(results_table,headers="firstrow", 6tablefmt='grid'))

files_to_plot = [
    '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Heavy/288/G18_s20222881550206_43.37_-123.25_114.tif',
    '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Heavy/132/G16_s20221322300206_35.84_-103.65_23.tif',
    '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Medium/149/G17_s20221491430321_33.11_-108.26_26.tif',
    '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Heavy/022/G17_s20220221350321_18.3_-71.3_100.tif',
    '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Light/274/G16_s20222742350202_34.04_-98.23_25.tif',
    '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Medium/086/G17_s20220861340321_35.53_-83.01_126.tif',
    '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Heavy/149/G17_s20221491250321_28.76_-107.5_52.tif',
    '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Heavy/270/G16_s20222702310206_60.57_-110.17_38.tif',
    "/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Medium/270/G16_s20222702310206_37.04_-121.21_33.tif",
    "/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/truth/2022/Medium/270/G18_s20222701530202_44.24_-122.74_18.tif"
]

### Setup samples for testing and plotting ###
samples_data_dict = {
    'data': [],
    'truth': []
}
for file_idx, fn in enumerate(tqdm(files_to_plot, desc="Processing files")):
    samples_data_dict['truth'].append(fn)
    samples_data_dict['data'].append(fn.replace('truth', 'data'))

print(samples_data_dict)

def plot_model_predictions(files_to_plot, input_list, ensemble=True, plot_individual_models=True):
    ### Setup models ###
    model_list = []
    loss_fn = nn.BCEWithLogitsLoss()
    for exp_num in tqdm(input_list, desc="Loading models"):
        _, _, model, _ = load_ckpt(exp_num)
        model_list.append(model)

    test_set = SmokeDataset(samples_data_dict, data_transforms)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, 
        batch_size=1, 
        shuffle=False, 
        drop_last=True,
    )

    ### Get all the predictions and iou scores from each model for each sample ###
    all_preds_probabilities = []; iou_list = []; all_ensemble_iou = []; all_ensemble_preds = []
    print(len(model_list))
    for model_ind, model in enumerate(model_list):
        print(f'================== MODEL {model_ind} RESULTS ==================')
        model.eval()
        torch.set_grad_enabled(False)
        for idx, data in enumerate(test_loader):
            batch_data, batch_labels, truth_fn = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)    
            # print('file:', truth_fn)
            
            iou_dict= {'high': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 
            'medium': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 
            'low': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}}

            pred = model(batch_data)
            pred = torch.sigmoid(pred)
            all_preds_probabilities.append(pred) # Save the preds in probability space for ensembling.
            pred = (pred > 0.5) * 1
            # pred = pred.squeeze(0).cpu().detach().numpy()
            # print('pred shape:', pred.shape)
            
            iou_dict= compute_iou(pred[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
            iou_dict= compute_iou(pred[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
            iou_dict= compute_iou(pred[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
            
            # Save the overall iou score for this model
            iou = get_iou_by_density(iou_dict)[3]
            # iou = iou.cpu().detach().numpy()
            iou_list.append(iou)

    print(len(all_preds_probabilities), len(iou_list))

    ### Get ensemble predictions and iou scores ###
    print('================== ENSEMBLE MODEL RESULTS ==================')
    for idx, data in enumerate(test_loader):
        batch_data, batch_labels, truth_fn = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        # print('file:', truth_fn)

        iou_dict= {'high': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 
            'medium': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 
            'low': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}}

        temp_preds = []
        # add each model's prediction to the list from the all preds list
        n_models =  len(model_list); n_samples = len(files_to_plot)
        for model_ind in range(n_models):
            temp_preds.append(all_preds_probabilities[n_samples * model_ind + idx])

        current_ensemble_preds = ens_probabilities_no_sigmoid(temp_preds)
        all_ensemble_preds.append(current_ensemble_preds)
        current_ensemble_iou = get_iou_by_density(compute_iou(current_ensemble_preds, batch_labels, 'high', iou_dict))[3]
        all_ensemble_iou.append(current_ensemble_iou)
    
    print(len(all_ensemble_preds), len(all_ensemble_iou))

    ### Plot preds ###
    print('================== PLOTTING RESULTS ==================')
    n_rows = 1
    n_cols = len(input_list) + (2 if ensemble else 1)
    # If the length of the input list is greater than 4, make the number of rows 2 and the number of cols in accordance
    if len(input_list) > 4 and plot_individual_models:
        n_rows = 2
        n_cols = len(input_list) // 2 + 1

    if not plot_individual_models:
        n_cols = 2
    colors = ['red', 'orange', 'yellow']
    titles = ['HMS Annotation']
    if plot_individual_models:
        for exp_num in input_list:
            model_title = '' #'exp' + exp_num
            for param in params_to_show_list:
                param_value = get_param_from_config(exp_num, param)
                if param == 'encoder': # get rid of the 'timm-' prefix at the start of timm-efficientnet-bx
                    split = param_value.split('-')
                    param_value = split[-2].capitalize() + ' ' + split[-1]
                if model_title == '':
                    model_title += f'{param_value}'
                else:
                    model_title += f'; {param_value}'
            titles.append(model_title)
    if ensemble and len(input_list) > 1:
        titles.append('Ensemble')

    for sample_idx, data_fn in enumerate(samples_data_dict['data']):
        truth_fn = samples_data_dict['truth'][sample_idx]
        print('file:', data_fn)

        # Setup figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=((n_cols*4, n_rows*4)))
        ax = axes.ravel() # Flatten axes for easier indexing

        # Load data
        RGB = skimage.io.imread(data_fn, plugin='tifffile')
        print('RBG Shape', RGB.shape)
        truths = skimage.io.imread(truth_fn, plugin='tifffile')
        print('Truths Shape', truths.shape)
        num_pixels = RGB.shape[1]
        X, Y = get_mesh(num_pixels)
        
        # Plot ground truth
        ax[0].imshow(RGB)
        for i in reversed(range(3)):
            ax[0].contour(X,Y,truths[:,:,i], levels=[.99], colors=[colors[i]])
        ax[0].set_title(r'$\mathbf{HMS}$' + ' ' + r'$\mathbf{Annotation}$', fontsize=12)

        if plot_individual_models:
            for model_idx, model in enumerate(model_list):
                temp_pred = all_preds_probabilities[n_samples * model_idx + sample_idx]
                temp_pred = (temp_pred > 0.5) * 1
                # print('temp_pred shape:', temp_pred.shape)

                # Plot individual model prediction
                ax[model_idx+1].imshow(RGB)
                for i in reversed(range(3)):
                    ax[model_idx+1].contour(X,Y,temp_pred[0,i,:,:].cpu(), levels=[.99], colors=[colors[i]])
                
                iou = iou_list[n_samples * model_idx + sample_idx]
                
                ax[model_idx+1].set_title(titles[model_idx+1], fontsize=12)
                ax[model_idx+1].set_xlabel(f'IoU: {iou:.3f}', fontsize=14)
            
        # Plot ensemble 
        if ensemble and len(model_list) > 1:
            ens_iou = all_ensemble_iou[sample_idx]
            ens_pred = all_ensemble_preds[sample_idx]

            ax[-1].imshow(RGB)
            for i in reversed(range(3)):
                ax[-1].contour(X,Y,ens_pred[0,i,:,:].cpu(), levels=[.99], colors=[colors[i]])
            
            # Calculate ensemble Iou
            ax[-1].set_title(r'$\mathbf{Ensemble}$', fontsize=12)
            ax[-1].set_xlabel(f'IoU: {ens_iou:.3f}', fontsize=16)
        
        # Clean up axes
        for a in ax.ravel():
            a.set_xticks([])
            a.set_yticks([])
        
        # Add title and save
        lat, lon = get_center_lat_lon(data_fn)
        plt.suptitle(get_datetime(data_fn.split('/')[-1]) + f' {lat}, {lon}'
                     , fontsize=14)
        plt.tight_layout()
        os.makedirs('multi_model_results', exist_ok=True)
        os.makedirs(f'multi_model_results/{sample_idx}', exist_ok=True)
        if plot_individual_models:
            save_fp = f'multi_model_results/{sample_idx}/final_{input}_{sample_idx}_w_members.png'
        else:
            save_fp = f'multi_model_results/{sample_idx}/final_{input}_{sample_idx}.png'
        plt.savefig(save_fp, bbox_inches='tight', dpi=300)  
        # os.makedirs('sample_results', exist_ok=True)
        # plt.savefig(f'sample_results/{input}_{sample_idx}.png', bbox_inches='tight', dpi=300)
        plt.close()

plot_model_predictions(files_to_plot, input_list, ensemble=(ensemble == 'T'))
plot_model_predictions(files_to_plot, input_list, ensemble=(ensemble == 'T'), plot_individual_models=False)
