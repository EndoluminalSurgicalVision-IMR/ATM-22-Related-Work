# -*- coding: utf-8 -*-

'''
Program :   Predict procedure.
Author  :   Minghui Zhang, Institute of Medical Robotics, Shanghai Jiao Tong University.
File    :   predict.py
Date    :   2022/8/1 16:28
Version :   V1.0
'''

import os
import sys
import argparse

import torch
import numpy as np
import SimpleITK as sitk

from model import UNet3D
from monai.transforms import (
    AddChannel,
    AsDiscrete,
    CastToType,
    KeepLargestConnectedComponent,
    ScaleIntensityRange,
    SqueezeDim,
    ToNumpy,
    ToTensor
)
from monai.inferers import sliding_window_inference


def getabspath(RelativePathList):
    length = len(RelativePathList)
    path = None
    for i in range(length - 1, 0, -1):
        if i == (length - 1):
            path = os.path.join(RelativePathList[i - 1], RelativePathList[i])
        else:
            path = os.path.join(RelativePathList[i - 1], path)
    return path


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    return numpyImage, numpyOrigin, numpySpacing


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def save_itk(image, filename, origin=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0]):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)


## Hyperparameters
in_channels = 1
out_channels = 2
init_fmaps_degree = 16
final_sigmoid = 1
depth = 128
height = 128
width = 128


class InnerTransform(object):
    def __init__(self):
        self.AddChannel = AddChannel()
        self.AsDiscrete = AsDiscrete(threshold=0.5)
        self.CastToNumpyUINT8 = CastToType(dtype=np.uint8)
        self.KeepLargestConnectedComponent = KeepLargestConnectedComponent(applied_labels=1, connectivity=3)
        self.ScaleIntensityRanger = ScaleIntensityRange(a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True)
        self.SqueezeDim = SqueezeDim()
        self.ToNumpy = ToNumpy()
        self.ToTensorFloat32 = ToTensor(dtype=torch.float)


InnerTransformer = InnerTransform()

parser = argparse.ArgumentParser('Baseline for Airway Tree Modeling (ATM22)')
parser.add_argument('-i', "--inputs", default='./inputs', type=str, help="input path of the CT images list")
parser.add_argument('-o', "--outputs", default='./outputs', type=str, help="output of the prediction results list")
args = parser.parse_args()

if __name__ == "__main__":
    with torch.no_grad():
        # load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = UNet3D(in_channels=in_channels,
                     out_channels=out_channels,
                     finalsigmoid=final_sigmoid,
                     fmaps_degree=init_fmaps_degree,
                     GroupNormNumber=4,
                     fmaps_layer_number=4,
                     layer_order='cip').to(device)
        net.eval()
        weight_path = 'weight.pth'
        net.load_state_dict(torch.load(weight_path))
        # ensure the output folder can be created
        mkdir(args.outputs)
        test_images_filelist = os.listdir(args.inputs)
        # prediction procedure
        for idx in range(0, len(test_images_filelist)):
            casename = test_images_filelist[idx]
            test_image_array, origin, spacing = load_itk_image(getabspath([args.inputs, casename]))
            test_image_array = InnerTransformer.AddChannel(test_image_array)
            test_image_array = InnerTransformer.ScaleIntensityRanger(test_image_array)
            test_image_array = InnerTransformer.AddChannel(test_image_array)
            test_image_tensor = InnerTransformer.ToTensorFloat32(test_image_array)
            test_image_tensor = test_image_tensor.to(device)
            pred = sliding_window_inference(inputs=test_image_tensor,
                                            roi_size=(depth, height, width),
                                            sw_batch_size=1,
                                            predictor=net,
                                            overlap=0.25,
                                            mode='gaussian',
                                            sigma_scale=0.125)
            pred = InnerTransformer.AsDiscrete(pred[:, 1, ...])
            pred = InnerTransformer.KeepLargestConnectedComponent(pred)
            pred = InnerTransformer.SqueezeDim(pred)
            pred = InnerTransformer.ToNumpy(pred)
            pred = InnerTransformer.CastToNumpyUINT8(pred)
            save_itk(pred, getabspath([args.outputs, casename]), origin, spacing)
