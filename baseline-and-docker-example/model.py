# -*- coding: utf-8 -*-

'''
Program :   Model file
Author  :   Minghui Zhang, Institute of Medical Robotics, Shanghai Jiao Tong University.
File    :   model.py
Date    :   2022/8/1 16:23
Version :   V1.0
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
import numpy as np


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]
    # return [32,64,128,256]


def create_conv(in_channels, out_channels, kernel_size, order='cri', GroupNumber=8, padding=1, stride=1, Deconv=False):
    """
    @Compiled by zmh
    create an ordered convlution layer for the UNet

    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param layer_order: the order of layer common match :
                        'cr'-> conv+relu
                        'crb'->conv+relu+batchnorm
                        'crg'->conv+relu+groupnorm(groupnorm number is designed)
                        ......
    :param GroupNumber:
    :param Padding:
    :return:
    """
    assert 'c' in order, 'Convolution must have a conv operation'
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'p':
            modules.append(('pReLU', nn.PReLU(num_parameters=out_channels)))
        elif char == 's':
            modules.append(('swish', swish()))
        elif char == 'm':
            modules.append(('Mish', Mish()))
        # elif char == 's':
        #     modules.append('Swish',nn.swish())
        elif char == 'c':
            if not Deconv:
                # add learnable bias only in the absence of gatchnorm/groupnorm
                bias = not ('g' in order or 'b' in order)
                modules.append(('conv', nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, padding=padding,
                                                  stride=stride)))
            else:
                bias = not ('g' in order or 'b' in order)
                modules.append(('convtranspose3d',
                                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias, padding=padding,
                                                   stride=stride)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            # number of groups must be less or equal the number of channels
            if out_channels < GroupNumber:
                GroupNumber = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=GroupNumber, num_channels=out_channels)))
        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                # affine true ---> with learnable parameters
                modules.append(('instancenorm', nn.InstanceNorm3d(in_channels, affine=True)))
            else:
                modules.append(('instancenorm', nn.InstanceNorm3d(out_channels, affine=True)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError("Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c','i']")
    return modules


# General SingleConv
class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm.
    The order of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers  default crb
        GroupNumber (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size, order='crg', GroupNumber=8):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, GroupNumber):
            self.add_module(name, module)


# General Doubleconv
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, encoder, order='crg', GroupNumber=8):
        super(DoubleConv, self).__init__()
        # Encoder
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if (conv1_out_channels < in_channels):
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        # Decoder
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # Conv1:
        self.add_module(name='Conv1', module=SingleConv(in_channels=conv1_in_channels,
                                                        out_channels=conv1_out_channels,
                                                        kernel_size=kernel_size,
                                                        order=order,
                                                        GroupNumber=GroupNumber))
        # Conv2:
        self.add_module(name='Conv2', module=SingleConv(in_channels=conv2_in_channels,
                                                        out_channels=conv2_out_channels,
                                                        kernel_size=kernel_size,
                                                        order=order,
                                                        GroupNumber=GroupNumber))


# SingleConvOnce
class SingleConvOnce(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, encoder, order='crg', GroupNumber=8):
        super(SingleConvOnce, self).__init__()
        # Conv1:
        self.add_module(name='Conv1', module=SingleConv(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        order=order,
                                                        GroupNumber=GroupNumber))

class UNet3D_Encoder(nn.Module):
    """
    Encode the network
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_kernelsize=(2, 2, 2),
                 pooling_type='max', apply_pooling=True, Basic_Module=DoubleConv, order='crg', GroupNumber=8):
        super(UNet3D_Encoder, self).__init__()
        assert pooling_type in ['max', 'avg'], 'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None
        if (Basic_Module == DoubleConv) or (Basic_Module == SingleConvOnce):
            self.basic_module = Basic_Module(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             encoder=True, order=order,
                                             GroupNumber=GroupNumber)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class UNet3D_Decoder(nn.Module):
    """
    Decode the network
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_kernelsize=(2, 2, 2),
                 Basic_Module=DoubleConv, order='crb', GroupNumber=8):
        super(UNet3D_Decoder, self).__init__()
        if (Basic_Module == DoubleConv or Basic_Module == SingleConvOnce):
            self.upsample = None
        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=False, order=order,
                                         GroupNumber=GroupNumber)

    def forward(self, encoder_feature, x):
        if self.upsample is None:
            # encoder's feature's ---> D*H*W
            output_size = encoder_feature.size()[2:]
            # encoder_feature,att_map = self.attention_module(encoder_feature,x)
            # x = F.interpolate(input=x,size=output_size,mode='nearest')
            x = F.interpolate(input=x, size=output_size, mode='trilinear', align_corners=True)
            x = torch.cat((encoder_feature, x), dim=1)
        x = self.basic_module(x)
        return x


class UNet3D(nn.Module):
    """
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    """

    def __init__(self, in_channels, out_channels, finalsigmoid, fmaps_degree, GroupNormNumber,
                 fmaps_layer_number, layer_order, **kwargs):
        super(UNet3D, self).__init__()
        # self.device = device
        assert isinstance(fmaps_degree, int), 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(fmaps_degree, fmaps_layer_number)
        # fmaps_list = [8,64,128,256]

        self.EncoderLayer1 = UNet3D_Encoder(in_channels=in_channels,
                                            out_channels=fmaps_list[0],
                                            apply_pooling=False,
                                            Basic_Module=DoubleConv,
                                            order=layer_order,
                                            GroupNumber=GroupNormNumber)
        self.EncoderLayer2 = UNet3D_Encoder(in_channels=fmaps_list[0],
                                            out_channels=fmaps_list[1],
                                            apply_pooling=True,
                                            Basic_Module=DoubleConv,
                                            order=layer_order,
                                            GroupNumber=GroupNormNumber)
        self.EncoderLayer3 = UNet3D_Encoder(in_channels=fmaps_list[1],
                                            out_channels=fmaps_list[2],
                                            apply_pooling=True,
                                            Basic_Module=DoubleConv,
                                            order=layer_order,
                                            GroupNumber=GroupNormNumber)
        self.EncoderLayer4 = UNet3D_Encoder(in_channels=fmaps_list[2],
                                            out_channels=fmaps_list[3],
                                            apply_pooling=True,
                                            Basic_Module=DoubleConv,
                                            order=layer_order,
                                            GroupNumber=GroupNormNumber)

        DecoderFmapList = list(reversed(fmaps_list))

        self.DecoderLayer1 = UNet3D_Decoder(in_channels=DecoderFmapList[0] + DecoderFmapList[1],
                                            out_channels=DecoderFmapList[1],
                                            Basic_Module=DoubleConv,
                                            order=layer_order,
                                            GroupNumber=GroupNormNumber)
        self.DecoderLayer2 = UNet3D_Decoder(in_channels=DecoderFmapList[1] + DecoderFmapList[2],
                                            out_channels=DecoderFmapList[2],
                                            Basic_Module=DoubleConv,
                                            order=layer_order,
                                            GroupNumber=GroupNormNumber)
        self.DecoderLayer3 = UNet3D_Decoder(in_channels=DecoderFmapList[2] + DecoderFmapList[3],
                                            out_channels=DecoderFmapList[3],
                                            Basic_Module=DoubleConv,
                                            order=layer_order,
                                            GroupNumber=GroupNormNumber)

        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1)

        if finalsigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        encoder_features = []
        x1 = self.EncoderLayer1(x)
        encoder_features.insert(0, x1)
        x2 = self.EncoderLayer2(x1)
        encoder_features.insert(0, x2)
        x3 = self.EncoderLayer3(x2)
        encoder_features.insert(0, x3)
        x4 = self.EncoderLayer4(x3)

        x = self.DecoderLayer1(encoder_features[0], x4)
        x = self.DecoderLayer2(encoder_features[1], x)
        x = self.DecoderLayer3(encoder_features[2], x)

        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x