import numpy as np
import os
from scipy import ndimage
import skimage.measure as measure
import nibabel
from skimage.morphology import skeletonize_3d


def large_connected_domain(label):
    cd, num = measure.label(label, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    # print(volume_sort)
    label = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    label = ndimage.binary_fill_holes(label)
    label = label.astype(np.uint8)
    return label


def skeleton_parsing(skeleton):
    # separate the skeleton
    neighbor_filter = ndimage.generate_binary_structure(3, 3)
    skeleton_filtered = ndimage.convolve(skeleton, neighbor_filter) * skeleton
    # distribution = skeleton_filtered[skeleton_filtered>0]
    # plt.hist(distribution)
    skeleton_parse = skeleton.copy()
    skeleton_parse[skeleton_filtered > 3] = 0
    con_filter = ndimage.generate_binary_structure(3, 3)
    cd, num = ndimage.label(skeleton_parse, structure=con_filter)
    # remove small branches
    for i in range(num):
        a = cd[cd == (i + 1)]
        if a.shape[0] < 5:
            skeleton_parse[cd == (i + 1)] = 0
    cd, num = ndimage.label(skeleton_parse, structure=con_filter)
    return skeleton_parse, cd, num


def tree_parsing_func(skeleton_parse, label, cd):
    # parse the airway tree
    edt, inds = ndimage.distance_transform_edt(1 - skeleton_parse, return_indices=True)
    tree_parsing = np.zeros(label.shape, dtype=np.uint16)
    tree_parsing = cd[inds[0, ...], inds[1, ...], inds[2, ...]] * label
    return tree_parsing


def loc_trachea(tree_parsing, num):
    # find the trachea
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((tree_parsing == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    # print(volume_sort)
    trachea = (volume_sort[-1] + 1)
    return trachea


def get_parsing(mask, refine=False):
    mask = (mask > 0).astype(np.uint8)
    mask = large_connected_domain(mask)
    skeleton = skeletonize_3d(mask)
    skeleton_parse, cd, num = skeleton_parsing(skeleton)
    tree_parsing = tree_parsing_func(skeleton_parse, mask, cd)
    return tree_parsing
