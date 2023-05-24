# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 19:23:00 2021

@author: Hao Zheng
"""
import numpy as np
import os
from scipy import ndimage
import SimpleITK as sitk

def neighbor_descriptor(label, filters):
    den = filters.sum()
    conv_label = ndimage.convolve(label.astype(np.float32), filters, mode='mirror')/den
    conv_label[conv_label==0] = 1
    conv_label = -np.log10(conv_label)
    return conv_label

def save_local_imbalance_based_weight(label_path, save_path):
    file_list = os.listdir(label_path)
    file_list.sort()
    for i in range(len(file_list)//5):
        label = np.load(os.path.join(label_path, file_list[5*i])) #load the binary labels
        filter0 = np.ones([7,7,7], dtype=np.float32)
        weight = neighbor_descriptor(label, filter0)       
        weight = weight*label
        #Here is constant weight. During training, varied weighted training is adopted.
        #weight = weight**np.random.random(2,3) * label + (1-label) in dataloader.
        weight = weight**2.5 
        weight = weight.astype(np.float16)
        save_name = save_path + file_list[5*i].split('_')[0] + "_weight.npy"
        np.save(save_name, weight)   

def save_lib_weight(label_path, save_path):
    file_list = os.listdir(label_path)
    file_list.sort()
    for f in file_list:
        print(f)
        label_Img = sitk.ReadImage(os.path.join(label_path, f))
        label = sitk.GetArrayFromImage(label_Img)
        name = f.split('.')[0]
        filter0 = np.ones([7, 7, 7], dtype=np.float32)
        weight = neighbor_descriptor(label, filter0)
        weight = weight * label
        # Here is constant weight. During training, varied weighted training is adopted.
        # weight = weight**np.random.random(2,3) * label + (1-label) in dataloader.
        # weight = weight ** 2.5
        weight = weight.astype(np.float16)
        save_name = os.path.join(save_path, name+'.npy')
        np.save(save_name, weight)

if __name__ == '__main__':
    save_lib_weight('./data/c_mask', './data/LIBBP/weight')

















