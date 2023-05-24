import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import cross_entropy, sigmoid, binary_cross_entropy
from WingsNet import WingsNet
from Data import CropSegData, SegValData, SegValCropData
from Data2 import AirwayHMData, OnlineHMData
import skimage.measure as measure
import nibabel
from skimage.morphology import skeletonize_3d
import SimpleITK as sitk
import bisect
import shutil

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = ((iflat) * tflat).sum()

    return 1 - ((2. * intersection + smooth) / ((iflat).sum() + (tflat).sum() + smooth))

def focal_loss(pred, label, gamma=2):
    N = pred.shape[0]
    pred = pred.view(N, -1)
    label = label.view(N, -1)
    probs = torch.sigmoid(pred)
    pt = torch.where(label == 1, probs, 1 - probs)
    ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred, label.float())
    loss = (torch.pow(1 - pt, gamma) * ce_loss)
    # loss = ce_loss
    loss = loss.mean()
    return loss

def Tversky_loss(pred, target):
    smooth = 1.0
    alpha = 0.05
    beta = 1 - alpha
    intersection = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()
    return 1 - (intersection + smooth) / (intersection + alpha * FP + beta * FN + smooth)

def general_union_loss(pred, target, dist):
    weight = dist * target + (1 - target)
    # when weight = 1, this loss becomes Root Tversky loss
    smooth = 1.0
    alpha = 0.1  # alpha=0.1 in stage1 and 0.2 in stage2
    beta = 1 - alpha
    sigma1 = 0.0001
    sigma2 = 0.0001
    weight_i = target * sigma1 + (1 - target) * sigma2
    intersection = (weight * ((pred + weight_i) ** 0.7) * target).sum()
    intersection2 = (weight * (alpha * pred + beta * target)).sum()
    return 1 - (intersection + smooth) / (intersection2 + smooth)

def general_union_loss_lib(pred, target, weight):
    smooth = 1.0
    alpha = 0.2  # alpha=0.1 in stage1 and 0.2 in stage2
    beta = 1 - alpha
    sigma1 = 0.0001
    sigma2 = 0.0001
    weight_i = target * sigma1 + (1 - target) * sigma2
    intersection = (weight * ((pred + weight_i) ** 0.7) * target).sum()
    intersection2 = (weight * (alpha * pred + beta * target)).sum()
    return 1 - (intersection + smooth) / (intersection2 + smooth)

def save_data_online(path, image, label, weight, names, limits=1500):
    name_list = os.listdir(os.path.join(path, 'image'))
    name_list.sort(key=lambda x:float(x.split('_')[0]))
    image = image.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    weight = weight.detach().cpu().numpy()
    dice_list = [float(n.split('_')[0]) for n in name_list]
    for i in range(image.shape[0]):
        if len(name_list) < limits:
            np.save(os.path.join(path, 'image', names[i]), image[i])
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))
            np.save(os.path.join(path, 'weight', names[i]), weight[i])
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
        else:
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            if index == 0: continue
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
            os.remove(os.path.join(path, 'image', name_list[0]))
            os.remove(os.path.join(path, 'label', name_list[0]))
            os.remove(os.path.join(path, 'weight', name_list[0]))
            name_list = name_list[1:]
            dice_list = dice_list[1:]
            np.save(os.path.join(path, 'image', names[i]), image[i])
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))
            np.save(os.path.join(path, 'weight', names[i]), weight[i])

def train():
    max_epoches = 50
    batch_size = 24
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    file_path = './data/base_dict.json'
    data_root = './data'

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    model = WingsNet(in_channel=2, n_classes=1)
    train_dataset = AirwayHMData(file_path=file_path, data_root=data_root, batch_size=batch_size, cube_size=128, ifbp=True)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=4,
                                   pin_memory=True, drop_last=True)
    valid_dataset = SegValCropData(file_path, data_root, batch_size=8, cube_size = 128, step = 64)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False, num_workers=4,
                                  pin_memory=True, drop_last=True)

    max_step = len(train_dataset) * max_epoches
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.0001, lr=0.01)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 40], gamma=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # resume
    weights_dict = torch.load(os.path.join('./saved_model', '0816_adddice' ,'wingsnet_37.pth'))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    for ep in range(0, max_epoches):
        if os.path.exists('./data/online_hardmining'):
            shutil.rmtree('./data/online_hardmining')
            os.mkdir('./data/online_hardmining')
            os.mkdir('./data/online_hardmining/image')
            os.mkdir('./data/online_hardmining/label')
            os.mkdir('./data/online_hardmining/weight')
        else:
            os.mkdir('./data/online_hardmining')
            os.mkdir('./data/online_hardmining/image')
            os.mkdir('./data/online_hardmining/label')
            os.mkdir('./data/online_hardmining/weight')

        for iter, pack in enumerate(train_data_loader):
            data = pack[0].float().cuda()
            data2 = pack[1].float().cuda()
            label = pack[2].float().cuda()
            weight = pack[3].float().cuda()

            data = data.transpose(0, 1)
            data2 = data2.transpose(0, 1)
            label = label.transpose(0, 1)
            weight = weight.transpose(0, 1)
            data = torch.cat([data, data2], dim=1)

            pred_en, pred_de = model(data)
            pred_en = torch.sigmoid(pred_en)
            pred_de = torch.sigmoid(pred_de)
            dice_loss_en = general_union_loss_lib(pred_en, label, weight)
            dice_loss_de = general_union_loss_lib(pred_de, label, weight)
            dice_loss_ori = dice_loss(pred_en, label) + dice_loss(pred_de, label)
            loss = dice_loss_de * 1 + dice_loss_en * 0.5 + dice_loss_ori * 0.5

            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            names = [general_union_loss_lib(pred_de[i], label[i], weight[i]) for i in range(data.shape[0])]
            names = [str(n.item())+'_'+str(iter)+'.npy' for n in names]
            save_data_online('./data/online_hardmining', data, label, weight, names,
                             limits=int(len(train_dataset)*batch_size*0.3))

            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset), '/', max_step,
                      'loss:', loss.item(), 'dice loss encode:', dice_loss_en.item(),
                      'dice loss decode:', dice_loss_de.item(), 'dice loss original:', dice_loss_ori.item())
            # break

        torch.cuda.empty_cache()
        # '''
        print('start online hard mining: ')
        hm_dataset = OnlineHMData(data_root='./data/online_hardmining', batch_size=24, rate=1.0)
        hm_dataloader = DataLoader(dataset=hm_dataset, batch_size=24, shuffle=True, num_workers=4,
                                   pin_memory=True, drop_last=True)
        for iter, pack in enumerate(hm_dataloader):
            data = pack[0].float().cuda()
            label = pack[1].float().cuda()
            weight = pack[2].float().cuda()
            pred_en, pred_de = model(data)
            dice_loss_ori = dice_loss(pred_en, label) + dice_loss(pred_de, label)
            pred_en = torch.sigmoid(pred_en)
            pred_de = torch.sigmoid(pred_de)
            dice_loss_all = general_union_loss_lib(pred_en, label, weight) * 0.5 + \
                            general_union_loss_lib(pred_de, label, weight)
            loss = dice_loss_ori * 0.5 + dice_loss_all
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset), '/', max_step,
                      'loss:', loss.item(), 'dice loss:', dice_loss_all.item(),
                      'dice loss:', dice_loss_ori.item())
        torch.cuda.empty_cache()
        # '''
        print('')
        validation(model, valid_dataloader, ep)
        print('')
        torch.save(model.module.state_dict(),
                   os.path.join('./saved_model', 'wingsnet_' + str(ep) + '.pth'))

def validation(model, valid_dataloader, epoch):
    model.train()
    # sliding window
    sens, pres, branches, dices = [], [], [], []
    last_name = ''
    flag = False
    with torch.no_grad():
        for i, (x, name, pos) in enumerate(valid_dataloader):
            name = name[0]
            # if name == 'ATM_093_0000': flag = True
            # if flag == False: continue
            if name != last_name:
                if last_name != '':
                    pred = pred / pred_num
                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    pred = np.squeeze(pred)
                    # np.save(os.path.join('./data/pred', last_name + '.npy'), pred)
                    # print(2 * (pred * label).sum() / ((pred + label).sum() + 1))
                    sen, pre, branch, dice = evaluation_case(pred, label, last_name)
                    sens.append(sen)
                    pres.append(pre)
                    branches.append(branch)
                    dices.append(dice)

                label = sitk.ReadImage(os.path.join('./data/c_mask', name+'.nii.gz'))
                label = sitk.GetArrayFromImage(label)
                pred = np.zeros(label.shape)
                pred = pred[np.newaxis, np.newaxis, ...]
                pred_num = np.zeros(pred.shape)
                last_name = name

            x = x.cuda()
            p0, p = model(x)
            p = torch.sigmoid(p)
            p = p.cpu().detach().numpy()
            pos = pos.numpy()
            for i in range(len(pos)):
                # print(pos)
                xl, xr, yl, yr, zl, zr = pos[i,0], pos[i,1], pos[i,2], pos[i,3], pos[i,4], pos[i,5]
                pred[0, :, xl:xr, yl:yr, zl:zr] += p[i]
                pred_num[0, :, xl:xr, yl:yr, zl:zr] += 1

        pred = pred / pred_num
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = np.squeeze(pred)
        # np.save(os.path.join('./data/pred', last_name + '.npy'), pred)
        # print(2 * (pred * label).sum() / ((pred + label).sum() + 1))
        # return
        sen, pre, branch, dice = evaluation_case(pred, label, last_name)
        sens.append(sen)
        pres.append(pre)
        branches.append(branch)
        dices.append(dice)

        sen_mean = np.mean(sens)
        sen_std = np.std(sens)
        pre_mean = np.mean(pres)
        pre_std = np.std(pres)
        branch_mean = np.mean(branches)
        branch_std = np.std(branches)
        dice_mean = np.mean(dices)
        dice_std = np.std(dices)
        print("len mean: %0.4f (%0.4f), branch: %0.4f (%0.4f), pre: %0.4f (%0.4f), dice: %0.4f (%0.4f)" % (
               sen_mean, sen_std, branch_mean, branch_std, pre_mean, pre_std, dice_mean, dice_std))
        line = "len mean: %0.4f (%0.4f), branch: %0.4f (%0.4f), pre: %0.4f (%0.4f), dice: %0.4f (%0.4f)" % ( \
               sen_mean, sen_std, branch_mean, branch_std, pre_mean, pre_std, dice_mean, dice_std)
        with open('./log.txt', 'a') as file:
            file.writelines(['epoch:' + str(epoch)+'\n', line+'\n', '\n'])

def evaluation_case(pred, label, name):
    parsing = sitk.ReadImage(os.path.join('./data', 'tree_parse', name + '.nii.gz'))
    parsing = sitk.GetArrayFromImage(parsing)
    if len(pred.shape) > 3:
        pred = pred[0]
    if len(label.shape) > 3:
        label = label[0]
    cd, num = measure.label(pred, return_num=True, connectivity=2)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    # print(volume_sort)
    large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)

    # skeleton = skeletonize_3d(label)
    skeleton = sitk.ReadImage(os.path.join('./data', 'skeleton', name + '.nii.gz'))
    skeleton = sitk.GetArrayFromImage(skeleton)
    skeleton = (skeleton > 0)
    skeleton = skeleton.astype('uint8')

    # print(pred.shape, label.shape, skeleton.shape)
    sen = (large_cd * skeleton).sum() / skeleton.sum()
    pre = (large_cd * label).sum() / large_cd.sum()

    num_branch = parsing.max()
    detected_num = 0
    # for j in range(num_branch):
    #     branch_label = ((parsing == (j + 1)).astype(int)) * skeleton
    #     if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8:
    #         detected_num += 1
    branch_label = parsing * skeleton
    branch_pred = branch_label * large_cd
    label_value_dic = dict(zip(*np.unique(branch_label, return_counts=True)))
    pred_value, pred_count = np.unique(branch_pred, return_counts=True)
    for j in range(len(pred_value)):
        if pred_value[j] == 0: continue
        if pred_count[j] / label_value_dic[pred_value[j]] >= 0.8:
            detected_num += 1

    branch = detected_num / num_branch

    dice = 2 * (pred * label).sum() / ((pred + label).sum() + 1)

    print(name, "Length: %0.4f" % (sen), "Precision: %0.4f" % (pre), "Branch: %0.4f" % (branch),
          "Dice: %0.4f" % (dice))
    return sen, pre, branch, dice

if __name__ == '__main__':
    train()





























