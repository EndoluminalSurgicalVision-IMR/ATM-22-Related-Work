import os
import time
import numpy as np
from importlib import import_module
import torch
from torch.utils.data import DataLoader
from Data import SegValData
import skimage.measure as measure
import nibabel
from skimage.morphology import skeletonize_3d


def network_prediction(data_path, save_path):
    casemodel = import_module('WingsNet')
    config2, case_net = casemodel.get_model()
    checkpoint = torch.load('WingsNet_GUL.ckpt')
    case_net.load_state_dict(checkpoint['state_dict'])
    val_path = data_path
    dataset = SegValData(val_path, train=False)
    val_loader_case = DataLoader(dataset, batch_size=1, shuffle=False)
    case_net = case_net.cuda()
    case_net.eval()
    save_path = save_path
    # sliding window
    cube_size = 128
    step = 64
    for i, (x, y, cb, patient) in enumerate(val_loader_case):
        pred = np.zeros(x.shape)
        pred_num = np.zeros(x.shape)
        x = x.cuda()
        xnum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 else \
            (x.shape[2] - cube_size) // step + 2
        ynum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 else \
            (x.shape[3] - cube_size) // step + 2
        znum = (x.shape[4] - cube_size) // step + 1 if (x.shape[4] - cube_size) % step == 0 else \
            (x.shape[4] - cube_size) // step + 2
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
                    p = torch.sigmoid(p)
                    p = p.cpu().detach().numpy()
                    pred[:, :, xl:xr, yl:yr, zl:zr] += p
                    pred_num[:, :, xl:xr, yl:yr, zl:zr] += 1

        pred = pred / pred_num
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = np.squeeze(pred)
        x = x.cpu().detach().numpy()
        x = np.squeeze(x)
        y = y.detach().numpy()
        y = np.squeeze(y)

        save_name_x = save_path + patient[0][0].split('_')[0] + '_img' + '.nii.gz'
        save_name_p = save_path + patient[0][0].split('_')[0] + '_pred' + '.nii.gz'
        save_name_y = save_path + patient[0][0].split('_')[0] + '_label' + '.nii.gz'

        img_nii = nibabel.Nifti1Image(x, np.eye(4))
        nibabel.save(img_nii, save_name_x)
        label_nii = nibabel.Nifti1Image(y, np.eye(4))
        nibabel.save(label_nii, save_name_y)
        pred_nii = nibabel.Nifti1Image(pred, np.eye(4))
        nibabel.save(pred_nii, save_name_p)


def evaluation(data_path, parsing_path):
    file_list = os.listdir(data_path)
    file_list.sort()
    file_list_parse = os.listdir(parsing_path)
    file_list_parse.sort()
    n = 3

    sens = []
    pres = []
    branches = []
    for i in range(len(file_list) // n):
        img = nibabel.load(os.path.join(data_path, file_list[n * i]))
        label = nibabel.load(os.path.join(data_path, file_list[n * i + 1]))
        pred = nibabel.load(os.path.join(data_path, file_list[n * i + 2]))
        parsing = nibabel.load(os.path.join(parsing_path, file_list_parse[6 * i + 4]))  # please refer to tree_parse.py
        img = img.get_data()
        label = label.get_data()
        pred = pred.get_data()
        parsing = parsing.get_data()

        cd, num = measure.label(pred, return_num=True, connectivity=1)
        volume = np.zeros([num])
        for k in range(num):
            volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
        volume_sort = np.argsort(volume)
        # print(volume_sort)
        large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)

        skeleton = skeletonize_3d(label)
        skeleton = (skeleton > 0)
        skeleton = skeleton.astype('uint8')

        sen = (large_cd * skeleton).sum() / skeleton.sum()
        sens.append(sen)

        pre = (large_cd * label).sum() / large_cd.sum()
        pres.append(pre)

        num_branch = parsing.max()
        detected_num = 0
        for j in range(num_branch):
            branch_label = ((parsing == (j + 1)).astype(np.uint8)) * skeleton
            if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8:
                detected_num += 1
        branch = detected_num / num_branch
        branches.append(branch)

        print(file_list[n * i].split('_')[0], "Length: %0.4f" % (sen), "Precision: %0.4f" % (pre),
              "Branch: %0.4f" % (branch))

    sen1_mean = np.mean(sens)
    sen1_std = np.std(sens)
    pre_mean = np.mean(pres)
    pre_std = np.std(pres)
    branch_mean = np.mean(branches)
    branch_std = np.std(branches)
    print("len mean: %0.4f (%0.4f), branch: %0.4f (%0.4f), pre: %0.4f (%0.4f)" % (
    sen1_mean, sen1_std, branch_mean, branch_std, pre_mean, pre_std))


if __name__ == '__main__':
    data_path = "/data_path/"
    save_path = "/save_path/"
    parsing_path = "/parsing_path/"
    network_prediction(data_path, save_path)
    evaluation(save_path, parsing_path)



