import torch
import numpy as np
import os
import nibabel
import skimage.measure as measure
from skimage.morphology import skeletonize_3d
from utils.tree_parse import get_parsing
import math

EPSILON = 1e-32


def compute_binary_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred) + EPSILON
    union = np.sum(y_true) + np.sum(y_pred) - intersection + EPSILON
    iou = intersection / union
    return iou

def evaluation_branch_metrics(fid,label, pred,refine=False):
    """
    :return: iou,dice, detected length ratio, detected branch ratio,
     precision, leakages, false negative ratio (airway missing ratio),
     large_cd (largest connected component)
    """
    # compute tree sparsing
    parsing_gt = get_parsing(label, refine)
    # find the largest component to locate the airway prediction
    cd, num = measure.label(pred, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    iou = compute_binary_iou(label, large_cd)
    flag=-1
    while iou < 0.1:
        print(fid," failed cases, require post-processing")
        large_cd = (cd == (volume_sort[flag-1] + 1)).astype(np.uint8)
        iou = compute_binary_iou(label, large_cd)
    skeleton = skeletonize_3d(label)
    skeleton = (skeleton > 0)
    skeleton = skeleton.astype('uint8')

    DLR = (large_cd * skeleton).sum() / skeleton.sum()
    precision = (large_cd * label).sum() / large_cd.sum()
    leakages = ((large_cd - label)==1).sum() / label.sum()

    num_branch = parsing_gt.max()
    detected_num = 0
    for j in range(num_branch):
        branch_label = ((parsing_gt == (j + 1)).astype(np.uint8)) * skeleton
        if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8:
            detected_num += 1
    DBR = detected_num / num_branch
    return iou, DLR, DBR, precision, leakages
