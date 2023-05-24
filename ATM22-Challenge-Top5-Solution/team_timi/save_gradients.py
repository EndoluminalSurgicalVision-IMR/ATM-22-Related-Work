# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 19:37:26 2021

@author: Hao Zheng
"""

import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import cross_entropy, sigmoid, binary_cross_entropy
from WingsNet import WingsNet
from Data import load_train_file
from Data2 import AirwayHMData
import skimage.measure as measure
import nibabel
from skimage.morphology import skeletonize_3d
import SimpleITK as sitk

torch.manual_seed(777) # cpu
torch.cuda.manual_seed(777) #gpu
np.random.seed(777) #numpy

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred
    tflat = target
    intersection = ((iflat) * tflat).sum()   
    return 1-((2. * intersection + smooth)/((iflat).sum() + (tflat).sum() + smooth))

def Tversky_loss(pred, target):
    smooth = 1.0
    alpha = 0.05
    beta = 1-alpha
    intersection = (pred*target).sum()
    FP = (pred*(1-target)).sum()
    FN = ((1-pred)*target).sum()
    return 1-(intersection + smooth)/(intersection + alpha*FP + beta*FN + smooth)

def root_Tversky_loss(pred, target, dist):
    alpha0 = 1
    beta0 = 1
    alpha = 0.05
    beta = 1 - alpha
    weight = (0.95*dist+0.05)*alpha0*target + beta0*(1-target)*dist
    #weight = 1
    smooth = 1.0
    sigma1 = 0.0001
    sigma2 = 0.0001
    weight_i = target*sigma1 + (1-target)*sigma2
    intersection = (weight*((pred+weight_i)**0.7)*target).sum()
    intersection2 = (weight*(alpha*pred + beta*target)).sum()
    return 1-(intersection + smooth)/(intersection2 + smooth)


def save_gradients(path, layer=0):   
    #load module
    casemodel = import_module('WingsNet')
    config2, case_net = casemodel.get_model()
    checkpoint = torch.load('005_0.ckpt') 
    case_net.load_state_dict(checkpoint['state_dict'])
    case_net = case_net.cuda()
    case_net.train()   
    
    grad_in = []
    def hook_fn_backward_output(module, grad_input, grad_output):
        grad_in.append(grad_output)
    
    def hook_fn_backward_input(module, grad_input, grad_output):
        if module.kernel_size[0]==3:
            grad_in.append(grad_output)
    
    for name, module in list(case_net.named_children()):
        if isinstance(module, nn.MaxPool3d):
            continue
        elif isinstance(module, nn.Conv3d):
            module.register_backward_hook(hook_fn_backward_output)
        elif isinstance(module, nn.Sigmoid):
            continue
        else:
            for name1, module1 in list(module.named_children()):
                if isinstance(module1, nn.Conv3d):
                    module1.register_backward_hook(hook_fn_backward_input)
    
    #load data
    file_list = os.listdir(path)
    file_list.sort()
    for idx in range(len(file_list)//6):
        img = np.load(os.path.join(path, file_list[6*idx]))
        label = np.load(os.path.join(path, file_list[6*idx+2]))
        weight = np.load(os.path.join(path, file_list[6*idx+1]))
        weight = weight**2.5
        weight = weight*label + (1-label)
        
        #calculate gradients
        img = img[np.newaxis,np.newaxis,...]
        label = label[np.newaxis,np.newaxis,...]
        weight = weight[np.newaxis,np.newaxis,...]
        x = torch.from_numpy(img.astype(np.float32)).cuda()
        y = torch.from_numpy(label.astype(np.float32)).cuda()
        w = torch.from_numpy(weight.astype(np.float32)).cuda()
        
        cube_size = 128
        step = 64
        pred = np.zeros(x.shape)
        pred_num = np.zeros(x.shape)
        grads = np.zeros(x.shape)
        grads_num = np.zeros(x.shape)
        #sliding window
        xnum = (x.shape[2]-cube_size)//step + 1 if (x.shape[2]-cube_size)%step==0 else (x.shape[2]-cube_size)//step + 2
        ynum = (x.shape[3]-cube_size)//step + 1 if (x.shape[3]-cube_size)%step==0 else (x.shape[3]-cube_size)//step + 2
        znum = (x.shape[4]-cube_size)//step + 1 if (x.shape[4]-cube_size)%step==0 else (x.shape[4]-cube_size)//step + 2
        for xx in range(xnum):
            xl = step*xx
            xr = step*xx + cube_size
            if xr > x.shape[2]:
                xr = x.shape[2]
                xl = x.shape[2]-cube_size
            for yy in range(ynum):
                yl = step*yy
                yr = step*yy + cube_size
                if yr > x.shape[3]:
                    yr = x.shape[3]
                    yl = x.shape[3] - cube_size
                for zz in range(znum):
                    zl = step*zz
                    zr = step*zz + cube_size
                    if zr > x.shape[4]:
                        zr = x.shape[4]
                        zl = x.shape[4] - cube_size
                    
                    x_input = x[:,:,xl:xr,yl:yr,zl:zr]
                    p0, p = case_net(x_input)
                    p_numpy = p.cpu().detach().numpy()
                    pred[:,:,xl:xr,yl:yr,zl:zr] += p_numpy
                    pred_num[:,:,xl:xr,yl:yr,zl:zr] += 1
                    
                    if label[:,:,xl:xr,yl:yr,zl:zr].sum()>0:
                        loss = root_Tversky_loss(p, y[:,:,xl:xr,yl:yr,zl:zr], w[:,:,xl:xr,yl:yr,zl:zr]) \
                        + 10*root_Tversky_loss(p0, y[:,:,xl:xr,yl:yr,zl:zr], w[:,:,xl:xr,yl:yr,zl:zr])
                        loss.backward()
                        grad_ec = grad_in[layer][0]
                        #print(grad_ec3.shape)
                        grad_ec = grad_ec.cpu().detach().numpy()
                        grad_ec = np.squeeze(grad_ec, 0)
                        grad_ec_abs = np.abs(grad_ec)
                        grad_ec_sum = np.sum(grad_ec_abs, axis=0)
                        grad_ec_norm = grad_ec_sum/grad_ec_sum.max()
                        grads[:,:,xl:xr,yl:yr,zl:zr] += grad_ec_norm
                        grads_num[:,:,xl:xr,yl:yr,zl:zr] += 1
                        grad_in = []
        
        
        pred = pred/pred_num
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0
        pred = np.squeeze(pred)
        
        grads_num[grads_num==0] = 1
        grads = grads/grads_num
        grads = np.squeeze(grads)
        
        #for i in range(20):
        #    grad_ec = grad_in[i][0]
        #    #print(grad_ec.shape)
        #    grad_ec = grad_ec.cpu().detach().numpy()
        #    grad_ec = np.squeeze(grad_ec, 0)
        #    grad_ec_abs = np.abs(grad_ec)
        #    grad_ec_sum = np.sum(grad_ec_abs, axis=0)
        #    grad_ec_norm = grad_ec_sum/grad_ec_sum.max()
        #    #grad_ec_norm = (grad_ec_sum>(grad_ec_sum.max()/5)).astype(np.uint8)
        #    #print(grad_ec_sum.max())
        #    
        #    down_sample = 128//grad_ec_norm.shape[1]
        #    ec_slice = grad_ec_norm[:,:,50//down_sample]
        #    if down_sample>1:
        #        ec_slice = ndimage.zoom(ec_slice, down_sample)
        #    save_name = "F:\\airway\\gradients\\figures\\WingsNet_0\\" + "hh%02d"%(i) + ".bmp"
        #    cv2.imwrite(save_name, 255*ec_slice)
        
        
        save_path = "data/"
        save_name_grad = save_path + file_list[6*idx+4].split('_')[0] + "_grad.nii.gz"
        save_name_label = save_path + file_list[6*idx+4].split('_')[0] + "_label.nii.gz"
        save_name_img = save_path + file_list[6*idx+4].split('_')[0] + "_img.nii.gz"
        save_name_pred = save_path + file_list[6*idx+4].split('_')[0] + "_pred.nii.gz"
        
        
        grad_nii = nibabel.Nifti1Image((grads).astype(np.float32), np.eye(4))
        nibabel.save(grad_nii, save_name_grad)
        label = label.squeeze()
        img = img.squeeze()
        label_nii = nibabel.Nifti1Image(label, np.eye(4))
        nibabel.save(label_nii, save_name_label)
        img_nii = nibabel.Nifti1Image(img, np.eye(4))
        nibabel.save(img_nii, save_name_img)
        pred_nii = nibabel.Nifti1Image(pred.astype(np.uint8), np.eye(4))
        nibabel.save(pred_nii, save_name_pred)

def process_img(data):
    data = data.astype(float)
    data2 = data.copy()
    data2[data2 > 500] = 500
    data2[data2 < -1000] = -1000
    data2 = (data2 + 1000) / 1500
    data[data > 1024] = 1024
    data[data < -1024] = -1024
    data = (data + 1024) / 2048
    return data, data2

def save_gradients_tw(layer=0):
    case_net = WingsNet(in_channel=2, n_classes=1)
    weights_dict = torch.load(os.path.join('./saved_model', 'wingsnet_4.pth'))
    case_net.load_state_dict(weights_dict, strict=False)
    # case_net = torch.nn.DataParallel(case_net).cuda()
    case_net.cuda()
    case_net.train()

    grad_in = []
    def hook_fn_backward_output(module, grad_input, grad_output):
        grad_in.append(grad_output)

    def hook_fn_backward_input(module, grad_input, grad_output):
        if module.kernel_size[0] == 3:
            grad_in.append(grad_output)

    for name, module in list(case_net.named_children()):
        if isinstance(module, nn.MaxPool3d):
            continue
        elif isinstance(module, nn.Conv3d):
            module.register_backward_hook(hook_fn_backward_output)
        elif isinstance(module, nn.Sigmoid):
            continue
        else:
            for name1, module1 in list(module.named_children()):
                if isinstance(module1, nn.Conv3d):
                    module1.register_backward_hook(hook_fn_backward_input)
    print(grad_in)
    # load data
    file_list = load_train_file('./data/base_dict.json', folder='0', mode=['train', 'val'])
    file_list.sort()
    for idx in range(len(file_list)):
        name = file_list[idx]
        img = sitk.ReadImage(os.path.join('./data/c_img', name+'.nii.gz'))
        img = sitk.GetArrayFromImage(img)
        label = sitk.ReadImage(os.path.join('./data/c_mask', name+'.nii.gz'))
        label = sitk.GetArrayFromImage(label)
        img, img2 = process_img(img)
        weight = np.load(os.path.join('./data/LIBBP/weight', name+'.npy'))

        # calculate gradients
        img = img[np.newaxis, np.newaxis, ...]
        img2 = img2[np.newaxis, np.newaxis, ...]
        label = label[np.newaxis, np.newaxis, ...]
        weight = weight[np.newaxis, np.newaxis, ...]
        x = torch.from_numpy(img.astype(np.float32)).cuda()
        x2 = torch.from_numpy(img2.astype(np.float32)).cuda()
        x = torch.cat((x, x2), dim=1)
        y = torch.from_numpy(label.astype(np.float32)).cuda()
        w = torch.from_numpy(weight.astype(np.float32)).cuda()

        cube_size = 128
        step = 64
        pred = np.zeros(y.shape)
        pred_num = np.zeros(y.shape)
        grads = np.zeros(y.shape)
        grads_num = np.zeros(y.shape)
        # sliding window
        xnum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 \
            else (x.shape[ 2] - cube_size) // step + 2
        ynum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 \
            else (x.shape[3] - cube_size) // step + 2
        znum = (x.shape[4] - cube_size) // step + 1 if (x.shape[4] - cube_size) % step == 0 \
            else (x.shape[4] - cube_size) // step + 2
        for xx in range(xnum):
            xl = step * xx
            xr = step * xx + cube_size
            if xr > x.shape[2]:
                xr = x.shape[2]
                xl = x.shape[2] - cube_size
            for yy in range(ynum):
                yl = step * yy
                yr = step * yy + cube_size
                if yr > x.shape[3]:
                    yr = x.shape[3]
                    yl = x.shape[3] - cube_size
                for zz in range(znum):
                    zl = step * zz
                    zr = step * zz + cube_size
                    if zr > x.shape[4]:
                        zr = x.shape[4]
                        zl = x.shape[4] - cube_size

                    x_input = x[:, :, xl:xr, yl:yr, zl:zr]
                    p0, p = case_net(x_input)
                    p_numpy = p.cpu().detach().numpy()
                    pred[:, :, xl:xr, yl:yr, zl:zr] += p_numpy
                    pred_num[:, :, xl:xr, yl:yr, zl:zr] += 1

                    if label[:, :, xl:xr, yl:yr, zl:zr].sum() > 0:
                        loss = root_Tversky_loss(p, y[:, :, xl:xr, yl:yr, zl:zr], w[:, :, xl:xr, yl:yr, zl:zr]) \
                               + 10 * root_Tversky_loss(p0, y[:, :, xl:xr, yl:yr, zl:zr], w[:, :, xl:xr, yl:yr, zl:zr])
                        loss.backward()
                        grad_ec = grad_in[layer][0]
                        # print(grad_ec3.shape)
                        grad_ec = grad_ec.cpu().detach().numpy()
                        grad_ec = np.squeeze(grad_ec, 0)
                        grad_ec_abs = np.abs(grad_ec)
                        grad_ec_sum = np.sum(grad_ec_abs, axis=0)
                        grad_ec_norm = grad_ec_sum / grad_ec_sum.max()
                        grads[:, :, xl:xr, yl:yr, zl:zr] += grad_ec_norm
                        grads_num[:, :, xl:xr, yl:yr, zl:zr] += 1
                        grad_in = []

        pred = pred / pred_num
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        grads_num[grads_num == 0] = 1
        grads = grads / grads_num

        print(name, 'dice', 2 * (pred * label).sum() / (pred + label).sum(), grads.shape)
        np.save(os.path.join('./data/LIBBP/grads', name+'.npy'), grads[0,0])
        pred_nii = nibabel.Nifti1Image(pred[0].astype(np.uint8), np.eye(4))
        nibabel.save(pred_nii, os.path.join(os.path.join('./data/LIBBP/preds', name+'.nii.gz')))

if __name__ == '__main__':
    save_gradients_tw(layer=0)















