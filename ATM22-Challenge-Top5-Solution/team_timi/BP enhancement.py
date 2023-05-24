# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:58:02 2021

@author: Hao Zheng
"""

import numpy as np
import os
import nibabel
from scipy import ndimage
from skimage.morphology import skeletonize_3d
import SimpleITK as sitk
from Data import load_train_file

def save_BP_weight_tw():
    file_list = load_train_file('./data/base_dict.json', folder='0', mode=['train', 'val'])
    file_list.sort()
    for i in range(len(file_list)):
        #load the label, predition and gradient
        name = file_list[i]
        label = sitk.ReadImage(os.path.join('./data/c_mask', name+'.nii.gz'))
        label = sitk.GetArrayFromImage(label)
        pred = nibabel.load(os.path.join('./data/LIBBP/preds', name+'.nii.gz'))
        pred = pred.get_data()[0]
        grad = np.load(os.path.join('./data/LIBBP/grads', name+'.npy'))
        
        fn = ((label.astype(np.float16) - pred)>0).astype(np.uint8)
        skeleton = skeletonize_3d(label)
        # grad_fn_skel = (1-grad)*fn*skeleton
        grad_fn_skel = fn*skeleton
        edt, inds = ndimage.distance_transform_edt(1-skeleton, return_indices=True)
        grad_wgt0 = grad_fn_skel[inds[0,...], inds[1,...], inds[2,...]] * label
        
        loc = (grad_wgt0>0).astype(np.uint8)
        f = loc * edt
        f = f * (1. - skeleton)
        maxf = np.amax(f)
        D = -((1./(maxf)) * f) + 1
        D = D * loc
        
        grad_wgt = (grad_wgt0**2)*(D**2)
        grad_wgt = grad_wgt.astype(np.float16)
        np.save(os.path.join('./data/LIBBP/distance_bp', name+'.npy'), grad_wgt)
        print(name)

def save_BP_weight(data_path, save_path):
    file_list = os.listdir(data_path)
    file_list.sort()
    for i in range(len(file_list) // 4):
        # load the label, predition and gradient
        label = nibabel.load(os.path.join(data_path + file_list[4 * i + 2]))
        pred = nibabel.load(os.path.join(data_path + file_list[4 * i + 3]))
        grad = nibabel.load(os.path.join(data_path + file_list[4 * i]))
        label = label.get_data()
        pred = pred.get_data()
        grad = grad.get_data()

        fn = ((label.astype(np.float16) - pred) > 0).astype(np.uint8)
        skeleton = skeletonize_3d(label)
        grad_fn_skel = (1 - grad) * fn * skeleton
        # grad_fn_skel = fn*skeleton
        edt, inds = ndimage.distance_transform_edt(1 - skeleton, return_indices=True)
        grad_wgt0 = grad_fn_skel[inds[0, ...], inds[1, ...], inds[2, ...]] * label

        loc = (grad_wgt0 > 0).astype(np.uint8)
        f = loc * edt
        f = f * (1. - skeleton)
        maxf = np.amax(f)
        D = -((1. / (maxf)) * f) + 1
        D = D * loc

        grad_wgt = (grad_wgt0 ** 2) * (D ** 2)
        grad_wgt = grad_wgt.astype(np.float16)
        save_name = save_path + file_list[4 * i + 1].split('_')[0] + "_dis2.npy"
        np.save(save_name, grad_wgt)

if __name__ == '__main__':
    save_BP_weight_tw()