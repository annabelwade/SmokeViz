import torch
import segmentation_models_pytorch as smp

def compute_iou(pred, true, level, iou_dict, print_ious=False, convert_to_classes=True):
    if convert_to_classes: # check if the preds were already converted to classes
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
    intersection = (pred + true == 2).sum()
    union = (pred + true >= 1).sum()
    iou = intersection / union
    # iou_dict[level]['prev_int'] = intersection
    # iou_dict[level]['prev_union'] = union
    if torch.isnan(iou) == False:
        iou_dict[level]['int'] += intersection
        iou_dict[level]['union'] += union
        if print_ious: print('{} density smoke gives: {} IoU'.format(level, iou))
        return iou_dict
    else:
        return iou_dict

def get_iou_by_density(iou_dict):
    try:
        high_iou = iou_dict['high']['int']/iou_dict['high']['union']
    except ZeroDivisionError:
        high_iou = float('nan') # np.nan
        
    try:
        med_iou = iou_dict['medium']['int']/iou_dict['medium']['union']
    except ZeroDivisionError:
        med_iou = float('nan')
        
    try:
        low_iou = iou_dict['low']['int']/iou_dict['low']['union']
    except ZeroDivisionError:
        low_iou = float('nan')
        
    try:
        iou = (iou_dict['high']['int'] + iou_dict['medium']['int'] + iou_dict['low']['int'])/(iou_dict['high']['union'] + iou_dict['medium']['union'] + iou_dict['low']['union']) # current overall IoU calculation
        
    except ZeroDivisionError:
        iou = float('nan')

    return [high_iou, med_iou, low_iou, iou]

def get_weighted_iou(iou_dict, dn_weights):
    [high_iou, med_iou, low_iou, _] = get_iou_by_density(iou_dict)
    weighted_iou = dn_weights[0]*high_iou + dn_weights[1]*med_iou + dn_weights[2]*low_iou
    return weighted_iou

def display_iou(iou_dict):
    [high_iou, med_iou, low_iou, iou] = get_iou_by_density(iou_dict)
    
    print('OVERALL HIGH DENSITY SMOKE GIVES: {} IoU'.format(high_iou))
    print('OVERALL MEDIUM DENSITY SMOKE GIVES: {} IoU'.format(med_iou))
    print('OVERALL LOW DENSITY SMOKE GIVES: {} IoU'.format(low_iou))
    print('OVERALL OVER ALL DENSITY GIVES: {} IoU'.format(iou))

    return [high_iou, med_iou, low_iou, iou]

def get_stats(pred, true, level, stats_dict, convert_to_classes=True, threshold=0.5):
    if convert_to_classes: # check if the preds were already converted to classes
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
    true = true.int()

    tp, fp, fn, tn = smp.metrics.get_stats(pred, true, threshold=threshold, mode='binary')
    stats_dict[level]['tp'] += tp.sum()
    stats_dict[level]['fp'] += fp.sum()
    stats_dict[level]['fn'] += fn.sum()
    stats_dict[level]['tn'] += tn.sum()
    return stats_dict

def compute_precision(stats_dict):
    try:
        high_precision = stats_dict['high']['tp']/(stats_dict['high']['tp'] + stats_dict['high']['fp'])
    except ZeroDivisionError:
        high_precision = float('nan')
        
    try:
        med_precision = stats_dict['medium']['tp']/(stats_dict['medium']['tp'] + stats_dict['medium']['fp'])
    except ZeroDivisionError:
        med_precision = float('nan')
        
    try:
        low_precision = stats_dict['low']['tp']/(stats_dict['low']['tp'] + stats_dict['low']['fp'])
    except ZeroDivisionError:
        low_precision = float('nan')
        
    try:
        precision = (stats_dict['high']['tp'] + stats_dict['medium']['tp'] + stats_dict['low']['tp'])/(stats_dict['high']['tp'] + stats_dict['medium']['tp'] + stats_dict['low']['tp'] + stats_dict['high']['fp'] + stats_dict['medium']['fp'] + stats_dict['low']['fp'])
    except ZeroDivisionError:
        precision = float('nan')
        
    precision_vals = [high_precision, med_precision, low_precision, precision]
    precision_vals = [item.item() for item in precision_vals]
    return precision_vals

def compute_recall(stats_dict):
    try:
        high_recall = stats_dict['high']['tp']/(stats_dict['high']['tp'] + stats_dict['high']['fn'])
    except ZeroDivisionError:
        high_recall = float('nan')
    
    try:    
        med_recall = stats_dict['medium']['tp']/(stats_dict['medium']['tp'] + stats_dict['medium']['fn'])
    except ZeroDivisionError:
        med_recall = float('nan')

    try:
        low_recall = stats_dict['low']['tp']/(stats_dict['low']['tp'] + stats_dict['low']['fn'])
    except ZeroDivisionError:
        low_recall = float('nan')

    try:
        recall = (stats_dict['high']['tp'] + stats_dict['medium']['tp'] + stats_dict['low']['tp'])/(stats_dict['high']['tp'] + stats_dict['medium']['tp'] + stats_dict['low']['tp'] + stats_dict['high']['fn'] + stats_dict['medium']['fn'] + stats_dict['low']['fn'])
    except ZeroDivisionError:
        recall = float('nan')

    recall_vals = [high_recall, med_recall, low_recall, recall]
    recall_vals = [item.item() for item in recall_vals]
    return recall_vals
    

