# encoding: utf-8
import sys
import os
import numpy as np
import torch.nn as nn
import pickle
import SimpleITK as sitk
from torch.nn.init import xavier_normal_, kaiming_normal_, constant_, normal_
from skimage import measure




smooth = 1.


def weights_init(net, init_type='normal'):
    """
    :param m: modules of CNNs
    :return: initialized modules
    """

    def init_func(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            if init_type == 'normal':
                normal_(m.weight.data)
            elif init_type == 'xavier':
                xavier_normal_(m.weight.data)
            else:
                kaiming_normal_(m.weight.data)
            if m.bias is not None:
                constant_(m.bias.data, 0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return


def load_pickle(filename='split_dataset.pickle'):
    """
    :param filename: pickle name
    :return: dictionary or list
    """
    with open(filename, 'rb') as handle:
        ids = pickle.load(handle)
    return ids


def save_pickle(dict, filename='split_dataset.pickle'):
    """
    :param dict: dictionary to be saved
    :param filename: pickle name
    :return: None
    """
    with open(filename, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def normalize_min_max(nparray):
    """
    :param nparray: inputs img (feature)
    :return: normalized nparray
    """
    nmin = np.amin(nparray)
    nmax = np.amax(nparray)
    norm_array = (nparray - nmin) / (nmax - nmin)
    return norm_array


def combine_total_avg(output, side_len, margin):
    """
    combine all things together and average overlapping areas of prediction
    curxinfo = [[curxdata, cursplitID, curnzhw, curshape, curorigin, curspacing]...]
    : param output: list of all coordinates and voxels of sub-volumes
    : param side_len: shape of the target volume
    : param margin: stride length of sliding window
    return: output_org, combined volume, original size
    return: curorigin, origin of CT
    return: curspacing, spacing of CT
    """
    curtemp = output[0]
    curshape = curtemp[3]
    curorigin = curtemp[4]
    curspacing = curtemp[5]
    #########################################################################
    nz, nh, nw = curtemp[2][0], curtemp[2][1], curtemp[2][2]
    [z, h, w] = curshape
    if type(margin) is not list:
        margin = [margin, margin, margin]

    splits = {}
    for i in range(len(output)):
        curinfo = output[i]
        curxdata = curinfo[0]
        cursplitID = int(curinfo[1])
        if not (cursplitID in splits.keys()):
            splits[cursplitID] = curxdata
        else:
            continue  # only choose one splits

    output = np.zeros((z, h, w), np.float32)

    count_matrix = np.zeros((z, h, w), np.float32)

    idx = 0
    for iz in range(nz + 1):
        for ih in range(nh + 1):
            for iw in range(nw + 1):
                sz = iz * side_len[0]
                ez = iz * side_len[0] + margin[0]
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
                split = splits[idx]
                ##assert (split.shape[0] == margin[0])
                ##assert (split.shape[1] == margin[1])
                ##assert (split.shape[2] == margin[2])
                # [margin[0]:margin[0] + side_len[0], margin[1]:margin[1] + \
                # side_len[1], margin[2]:margin[2] + side_len[2]]
                output[sz:ez, sh:eh, sw:ew] += split
                count_matrix[sz:ez, sh:eh, sw:ew] += 1
                idx += 1

    output = output / count_matrix
    output_org = output
    # output_org = output[:zorg, :horg, :worg]
    ##min_value = np.amin(output_org.flatten())
    ##max_value = np.amax(output_org.flatten())
    ##assert (min_value >= 0 and max_value <= 1)
    return output_org, curorigin, curspacing


def combine_total(output, side_len, margin):
    """
    combine all things together without average overlapping areas of prediction
    curxinfo = [[curxdata, cursplitID, curnzhw, curshape, curorigin, curspacing]...]
    : param output: list of all coordinates and voxels of sub-volumes
    : param side_len: shape of the target volume
    : param margin: stride length of sliding window
    return: output_org, combined volume, original size
    return: curorigin, origin of CT
    return: curspacing, spacing of CT
    """
    curtemp = output[0]
    curshape = curtemp[3]
    curorigin = curtemp[4]
    curspacing = curtemp[5]

    nz, nh, nw = curtemp[2][0], curtemp[2][1], curtemp[2][2]
    [z, h, w] = curshape
    #### output should be sorted
    if type(margin) is not list:
        margin = [margin, margin, margin]
    splits = {}
    for i in range(len(output)):
        curinfo = output[i]
        curxdata = curinfo[0]
        cursplitID = int(curinfo[1])
        splits[cursplitID] = curxdata

    output = -1000000 * np.ones((z, h, w), np.float32)

    idx = 0
    for iz in range(nz + 1):
        for ih in range(nh + 1):
            for iw in range(nw + 1):
                sz = iz * side_len[0]
                ez = iz * side_len[0] + margin[0]
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
                split = splits[idx]
                ##assert (split.shape[0] == margin[0])
                ##assert (split.shape[1] == margin[1])
                ##assert (split.shape[2] == margin[2])
                output[sz:ez, sh:eh, sw:ew] = split
                idx += 1
    output_org = output
    # output_org = output[:z, :h, :w]
    ##min_value = np.amin(output_org.flatten())
    ##assert (min_value >= -1000000)
    return output_org, curorigin, curspacing


def save_itk(image, origin, spacing, filename):
    """
    :param image: images to be saved
    :param origin: CT origin
    :param spacing: CT spacing
    :param filename: save name
    :return: None
    """
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


def load_itk_image(filename):
    """
    :param filename: CT name to be loaded
    :return: CT image, CT origin, CT spacing
    """
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def lumTrans(img, window_width=1600, window_level=-600, ret_type="uint8"):
    """
    :param img: CT image
    :param window_width: int, window width for
    :param window_level: CT image
    :return: Hounsfield Unit window clipped and normalized
    """
    lungwin = np.array([window_level - window_width / 2., window_level + window_width / 2.])
    # the upper bound 400 is already ok
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1

    if ret_type == "float":
        norm_img = (newimg.astype("float") - np.mean(newimg)) / np.std(newimg)
    else:
        newimg = (newimg * 255).astype('uint8')
    return newimg










def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_lung_extendbox(mask, margin=(5, 20, 20)):
    zz, yy, xx = np.where(mask)
    min_box = np.max([[0, zz.min() - margin[0]],
                      [0, yy.min() - margin[1]],
                      [0, xx.min() - margin[2]]], axis=1, keepdims=True)

    max_box = np.min([[mask.shape[0], zz.max() + margin[0]],
                      [mask.shape[1], yy.max() + margin[1]],
                      [mask.shape[2], xx.max() + margin[2]]], axis=1, keepdims=True)

    box = np.concatenate([min_box, max_box], axis=1)

    return box





def keep_largest_connected_component(mask, bg=0):
    """
    :brief 获取最大连通域
    :param[in] mask: ndarray, the binary mask with shape in order [z, y, x]
    :param[in] bg: int, the value of background, defaults, 0
    :return: the mask contain the max connected component
    """
    mask = mask.copy()
    labels = measure.label(mask)
    vals, counts = np.unique(labels, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        max_label = vals[np.argmax(counts)]
        mask[labels != max_label] = 0
        return mask.astype('bool').astype('uint8')
    else:
        return np.zeros_like(mask, dtype='uint8')

if __name__ == '__main__':
    data_path = r'D:\work\project\working\save_airway\val020\ATM_644_0000-pred.nii.gz'
    arr,_,_ = load_itk_image(data_path)
    arr = keep_largest_connected_component(arr)

