# -*- coding: utf-8 -*-

'''
Program :   Predict procedure.
Author  :   Minghui Zhang, Institute of Medical Robotics, Shanghai Jiao Tong University.
File    :   predict.py
Date    :   2022/8/1 16:28
Version :   V1.0
'''

import os
import argparse

import torch
import numpy as np
import SimpleITK as sitk

from model import UNet3D
from utils import lumTrans, load_itk_image, save_itk, compute_lung_extendbox, keep_largest_connected_component, mkdir
from segment_lung_mask import segment_lung_mask
import torch.nn.functional as F

out_channels = 2
cubesize = [128, 128, 128]
CUBE_SIZE = [128, 128, 128]  # 和下方网络定义的尺寸保持一致
SLIDE_SIZE = [64, 64, 64]  # 移动步长，有overlap


def load_model(model_file):
    model = torch.load(r'model/airway_seg.pth')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model


def get_model_input(ct_array, resolution):
    wc, ww = -600, 1600
    ct_array = (ct_array - wc) / ww
    sample_list = []
    z, y, x = ct_array.shape
    window = (-5, -2, 0, 2, 5)
    for slice_index in range(z):
        sample = np.zeros([len(window), y, x], 'float32')
        for idx in range(len(window)):
            slice_id = slice_index + int(window[idx] / resolution[0])
            if slice_id >= 0 and slice_id < z:
                sample[idx, :, :] = ct_array[slice_id, :, :]
        sample_list.append(sample)
    return sample_list


def predict(test_model, sample_list):
    sample_array = np.stack(sample_list, axis=0)  # z * 5 * y * x
    batch_size = 8
    prediction_list = []
    index = 0
    soft_max = torch.nn.Softmax(dim=1)
    test_model.eval()
    with torch.no_grad():
        while index < len(sample_list):
            index_end = index + batch_size
            if index_end >= len(sample_list):
                index_end = len(sample_list)
            inputs = torch.from_numpy(sample_array[index: index_end, :, :, :]).cuda()
            prediction = test_model(inputs)
            prediction = soft_max(prediction)
            prediction = prediction.cpu().numpy()  # batch_size * 2 * y * x
            prediction_list.append(prediction)
            index = index_end
    prediction_array = np.concatenate(prediction_list, axis=0)  # z * 2 * y * x
    lung_mask = np.array(prediction_array[:, 1, :, :] > 0.5, 'float32')
    return lung_mask


class SplitComb():
    def __init__(self, side_len=[80, 192, 304], margin=60):
        """
        :param side_len: list of inputs shape, default=[80,192,304] \
        :param margin: sliding stride, default=[60,60,60]
        """
        self.side_len = side_len
        self.margin = margin

    def split_id(self, data):
        """
        :param data: target data to be splitted into sub-volumes, shape = (D, H, W) \
        :return: output list of coordinates for the cropped sub-volumes, start-to-end
        """
        side_len = self.side_len
        margin = self.margin

        if type(margin) is not list:
            margin = [margin, margin, margin]

        splits = []
        z, h, w = data.shape

        nz = int(np.ceil(float(z - margin[0]) / side_len[0]))
        nh = int(np.ceil(float(h - margin[1]) / side_len[1]))
        nw = int(np.ceil(float(w - margin[2]) / side_len[2]))

        assert (nz * side_len[0] + margin[0] - z >= 0)
        assert (nh * side_len[1] + margin[1] - h >= 0)
        assert (nw * side_len[2] + margin[2] - w >= 0)

        nzhw = [nz, nh, nw]
        self.nzhw = nzhw

        pad = [[0, nz * side_len[0] + margin[0] - z],
               [0, nh * side_len[1] + margin[1] - h],
               [0, nw * side_len[2] + margin[2] - w]]
        orgshape = [z, h, w]

        idx = 0
        for iz in range(nz + 1):
            for ih in range(nh + 1):
                for iw in range(nw + 1):
                    sz = iz * side_len[0]  # start
                    ez = iz * side_len[0] + margin[0]  # end
                    sh = ih * side_len[1]
                    eh = ih * side_len[1] + margin[1]
                    sw = iw * side_len[2]
                    ew = iw * side_len[2] + margin[2]
                    if ez > z:
                        sz = z - margin[0]
                        ez = z
                    if eh > h:
                        sh = h - margin[1]
                        eh = h
                    if ew > w:
                        sw = w - margin[2]
                        ew = w
                    idcs = [[sz, ez], [sh, eh], [sw, ew], idx]
                    splits.append(idcs)
                    idx += 1
        splits = np.array(splits)
        # split size
        return splits, nzhw, orgshape


if __name__ == "__main__":
    input_dir = r''
    predict_dir = r''

    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)

    model_file = 'airway_seg.pth'
    model = load_model(model_file)

    print("model loaded successfully!")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for ct_file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, ct_file)
        dataname = ct_file.split('\.')[0]

        img_arr, origin, spacing = load_itk_image(input_file)
        lung_mask = segment_lung_mask(img_arr)

        box = compute_lung_extendbox(lung_mask)
        dcm_img = img_arr[box[0, 0]: box[0, 1], box[1, 0]: box[1, 1], box[2, 0]: box[2, 1]]
        dcm_img = lumTrans(dcm_img)

        # do inference
        split = SplitComb(SLIDE_SIZE, CUBE_SIZE)
        splits, nzhw, orgshape = split.split_id(dcm_img)

        batch = []
        split_coord = []
        [z, h, w] = orgshape
        output = np.zeros((z, h, w), np.float32)

        with torch.no_grad():
            for i in range(len(splits)):
                cursplit = splits[i]
                curcube = dcm_img[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
                          cursplit[2][0]:cursplit[2][1]]
                curcube = (curcube.astype(np.float32)) / 255.0

                # calculate the coordinate for coordinate-aware convolution
                start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
                normstart = ((np.array(start).astype('float') / np.array(orgshape).astype('float')) - 0.5) * 2.0
                crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
                stride = 1.0
                normsize = (np.array(crop_size).astype('float') / np.array(orgshape).astype('float')) * 2.0
                xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
                                         np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
                                         np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
                                         indexing='ij')
                coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype(
                    'float')
                curcube = curcube[np.newaxis, ...]
                batch.append([curcube, coord])
                split_coord.append(cursplit)
                if len(batch) < 2 and i != len(splits) - 1:
                    continue

                input_cube = torch.from_numpy(np.array([batch[i][0] for i in range(len(batch))])).float()
                input_coord = torch.from_numpy(np.array([batch[i][1] for i in range(len(batch))])).float()
                if torch.cuda.is_available():
                    # input_cube = input_cube.cuda()
                    input_cube = input_cube.to(device)
                    # input_coord = input_coord.cuda()
                    input_coord = input_coord.to(device)

                preds = model(input_cube, input_coord)  # (B, 3, D, H, W), 0=backgroud, 1=artery, 2=vein
                preds = F.softmax(preds, dim=1)
                outdatabw = np.argmax(preds.cpu().data.numpy(), axis=1)  # (B, D, H, W)

                for k in range(len(batch)):
                    cursplit = split_coord[k]
                    output[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
                    cursplit[2][0]:cursplit[2][1]] = outdatabw[k]

                batch = []
                split_coord = []

        torch.cuda.empty_cache()

        ret_mask = np.zeros_like(img_arr)
        ret_mask[box[0, 0]: box[0, 1], box[1, 0]: box[1, 1], box[2, 0]: box[2, 1]] = output
        ret_mask = keep_largest_connected_component(ret_mask)

        save_itk(ret_mask, origin, spacing, os.path.join(predict_dir, dataname + '.nii.gz'))

        print("segmentation is generated successfully!", dataname)
