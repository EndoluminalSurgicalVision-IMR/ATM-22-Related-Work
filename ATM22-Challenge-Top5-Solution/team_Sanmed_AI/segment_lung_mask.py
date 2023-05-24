import numpy as np
import SimpleITK as sitk
import os
from scipy import ndimage
from skimage import measure
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure, binary_opening, binary_erosion



def is_backgroud(label_region, origin_mask, k_num=5, margin=2):
    zz, yy, xx = np.where(label_region)
    z_length = zz.max() - zz.min() + 1
    y_length = yy.max() - yy.min() + 1
    x_length = xx.max() - xx.min() + 1
    # z_length, y_length, x_length = label_region.shape
    # 取靠近中间的5张（奇数张）来判断是否局部肺，少数服从多数
    start_index = z_length / 2 - (k_num / 2) * margin
    end_index = z_length / 2 + (k_num / 2) * margin + 1  # range是左开右闭区间
    # 异常值检测，防止超出数据范围，超出情况说明slice张数特别少，直接取中间张作为判断依据
    if start_index < 0 or end_index > z_length:
        start_index = z_length / 2
        end_index = z_length / 2 + 1  # range是左开右闭区间
        margin = 1

    selected_slice_index = list(range(start_index, end_index, margin))
    selected_slice_index_is_background = [False] * k_num
    for idx, index in enumerate(selected_slice_index):
        # 移除一些不连通噪声，保留胸壁，开操作去除背板以免造成触边误判
        y_region, x_region = np.where(label_region[index])
        # 背景区域一定是贯穿始终的，所以缺失则默认非背景，做跳过处理
        if len(y_region) and len(x_region):
            y_region_length = y_region.max() - y_region.min() + 1
            x_region_length = x_region.max() - x_region.min() + 1
        else:
            continue

        filled_region = ndi.binary_fill_holes(label_region[index])
        vessel_like = np.logical_xor(filled_region, origin_mask[index])
        # vessel_like = np.logical_xor(filled_region, label_region[index])
        y_vessel, x_vessel = np.where(vessel_like)
        # 肺内区域一定是有血管造成的孔洞的，所以没有血管就肯定是背景
        if len(y_vessel) and len(x_vessel):
            y_vessel_length = y_vessel.max() - y_vessel.min() + 1
            x_vessel_length = x_vessel.max() - x_vessel.min() + 1
        else:
            selected_slice_index_is_background[idx] = True
            continue

        label = measure.label(vessel_like)
        # 肺区域的话血管较多，连通域肯定不止3个，而且上下左右分布均匀，3只在此处使用，故写作hard code
        if label.max() < 3:
            selected_slice_index_is_background[idx] = True
            continue

        # 有背板也肯定是背景，背板特征
        if y_vessel.min() > 0.5 * (y_region.min() + y_region.max()) and y_vessel_length < 0.33 * y_region_length:
            selected_slice_index_is_background[idx] = True
            continue

    # 少数服从多数投票，超过半数slice为局部肺，则返回True
    if sum(selected_slice_index_is_background) > k_num / 2:
        return True
    return False


def is_local_lung(chest_mask, k_num=5, margin=2, min_area=2000):
    """
    :brief 通过判断胸腔是否封闭来判断是否局部肺
    :param[in] chest_mask: np.ndarray, the mask of chest
    :param[in] k_num: int, uneven number, like the k in KNN, the number of sampled slice for judge
    :param[in] margin: int, the sample step for the middle slice
    :param[in] min_area: int, about the area of trachea
    :return: bool, true means the dcm_img is local lung.
    """
    z_length, y_length, x_length = chest_mask.shape
    # 取靠近中间的5张（奇数张）来判断是否局部肺，少数服从多数
    start_index = z_length / 2 - (k_num / 2) * margin
    end_index = z_length / 2 + (k_num / 2) * margin + 1  # range是左开右闭区间
    # 异常值检测，防止超出数据范围，超出情况说明slice张数特别少，直接取中间张作为判断依据
    if start_index < 0 or end_index > z_length:
        start_index = z_length / 2
        end_index = z_length / 2 + 1  # range是左开右闭区间
        margin = 1

    selected_slice_index = list(range(start_index, end_index, margin))
    # 默认都不为局部肺
    selected_slice_index_is_local = [False] * k_num
    for idx, index in enumerate(selected_slice_index):
        # 移除一些不连通噪声，保留胸壁，开操作去除背板以免造成触边误判
        chest_mask[index] = binary_opening(chest_mask[index], iterations=2)
        label_image = measure.label(chest_mask[index])
        regions_image = measure.regionprops(label_image)
        max_area, seq = 0, 0
        for region in regions_image:
            if region.area > max_area:
                max_area = region.area
                seq = region.label
        # 这里两种同等效果赋值方式的速度好像有挺大差异，待测
        # mask[iz] = label_image == seq
        chest_mask[index] = np.in1d(label_image, [seq]).reshape(label_image.shape)

        filled_chest = ndi.binary_fill_holes(chest_mask[index])
        # 如果胸腔封闭则填孔后的面积增加量小于5 * min_area（经验来看中间层面的肺比5个气管面积大）
        max_bronchi_area = 5 * min_area
        if np.sum(filled_chest) - np.sum(chest_mask[index]) < max_bronchi_area:
            selected_slice_index_is_local[idx] = True
            continue
        else:
            label = measure.label(filled_chest ^ chest_mask[index])
            vals, counts = np.unique(label, return_counts=True)
            counts = sorted(counts[vals != 0].tolist(), reverse=True)
            # 保险判断，如果胸腔中有两个以上较大连通域，则肯定为全肺视角，不需要进行触边判断
            if label.max() >=2 and counts[0] > max_bronchi_area and counts[1] > max_bronchi_area:
                continue

        # 或者胸腔四周都触边，可认为是局部肺，这种case是单边肺，中间层面包含完整的半肺，其它层面是局部肺
        # 部分case四个角为圆环伪影，导致了误判，采用保留最大连通域后再进行判断
        yy, xx = np.where(filled_chest)
        border = 10  # 10个像素贴边即为触边
        if (yy.min() - 0 < border or y_length-1 - yy.max() < border) and (xx.min() - 0 < border or x_length-1-xx.max() < border):
            selected_slice_index_is_local[idx] = True
            continue

    # 少数服从多数投票，超过半数slice为局部肺，则返回True
    if sum(selected_slice_index_is_local) > k_num / 2:
        return True
    return False


def largest_one_or_two_label_volume(mask, origin_mask, bg=-1, is_half=False, is_local=False, is_circle=False):
    """
    :brief 获取连通域标记值的最大的两个标记区域的值
    :param[in] mask: ndarray, 连通域标记的labels, 一般由measure.label产生
    :param[in] bg: int, 背景值，默认为-1
    :return: 保留连通域后的mask及连通域个数
    """
    ret_mask = np.zeros_like(mask)
    labels = measure.label(mask, background=0)
    vals, counts = np.unique(labels, return_counts=True)
    counts = counts[vals != bg].astype('float')
    vals = vals[vals != bg]

    # 如果没有开肺定位，则默认按半肺的情况处理，有时间消耗
    # if not PREPROCESS_USE_SLICE_LOCATOR:
    is_half = True

    order = np.argsort(counts)[::-1].tolist()  # 连通域从大到小的索引

    # 局部肺封边可能导致背景出现，进行一个背景滤除，防止背景成为最大连通域影响后续的判断
    minimal_lung_region = 10000  # 该面积约为5个气管，小于这个体积的不当做肺，极小的局部肺或者术后的极小肺
    while len(order)>=2 and (is_local or is_circle):
        max_region = labels == vals[order[0]]
        # 确保最大的连通域不是背景就行
        if np.sum(max_region) > minimal_lung_region and is_backgroud(max_region, origin_mask):
            order.pop(0)
        else:
            break

    # 在完整的序列中由于气管的存在会出现只有一个连通域的情况
    if len(order) == 1 or not is_half:
        ret_mask = labels == vals[order[0]]
        return ret_mask.astype('uint8'), 1
    elif len(order) >= 2:
        for i in range(1, len(order)):
            if counts[order[i]] < minimal_lung_region:
                ret_mask = labels == vals[order[0]]
                return ret_mask.astype('uint8'), 1
            # 正常肺组织大小不会相差5倍
            if not is_local and counts[order[0]] / counts[order[i]] > 5.0:
                continue
            # 局部肺相差10倍就舍弃，说明另外一部分肺很小，为非主要区域
            if is_local and counts[order[0]] / counts[order[i]] > 5.0:
                continue
            else:
                # 不完整的序列没有气管会导致直接出现有两个半肺各自形成连通域，简单通过比较两个半肺的大小和Z轴位置来进行取舍
                # 此处经验值认为两个半肺大小差不会超过5倍，若超过5倍可能是由于手术等原因造成的，直接返回两个连通域，后续会做判断
                # 通过判断两个最大连通域的Z轴有无交叉来决定是否保留以防噪声面积过大
                # 如果进入到该逻辑，必定有i == 1, 最大连通域只需计算一次，进入到这个逻辑会增加耗时，但是很少case能进入这个逻辑
                if i == 1:
                    z_first, y_first, x_first = np.where(labels == vals[order[0]])
                    z_first_start, z_first_end = z_first.min(), z_first.max()
                    z_first_length = z_first_end - z_first_start + 1

                    y_first_start, y_first_end = y_first.min(), y_first.max()
                    y_first_length = y_first_end - y_first_start + 1

                    x_first_start, x_first_end = x_first.min(), x_first.max()
                    x_first_length = x_first_end - x_first_start + 1

                z_second, y_second, x_second = np.where(labels == vals[order[i]])
                z_second_start, z_second_end = z_second.min(), z_second.max()
                z_second_length = z_second_end - z_second_start + 1

                y_second_start, y_second_end = y_second.min(), y_second.max()
                y_second_length = y_second_end - y_second_start + 1

                x_second_start, x_second_end = x_second.min(), x_second.max()
                x_second_length = x_second_end - x_second_start + 1

                # 两个Z轴区间无交叉，或者区间长度差超过3倍，则pass该连通域，
                if z_second_start > z_first_end or z_second_end < z_first_start or \
                        float(z_first_length) / z_second_length > 3.0 or float(z_second_length) / z_first_length > 3.0:
                    if i == len(order) - 1:
                        ret_mask = labels == vals[order[0]]
                        return ret_mask.astype('uint8'), 1
                    continue

                # 两个Y轴区间无交叉，或者区间长度差超过3倍（左右对称的双肺高度一般为1：1），则pass该连通域，
                if y_second_start > y_first_end or y_second_end < y_first_start or \
                        float(y_first_length) / y_second_length > 3.0 or float(y_second_length) / y_first_length > 3.0:
                    if i == len(order) - 1:
                        ret_mask = labels == vals[order[0]]
                        return ret_mask.astype('uint8'), 1
                    continue

                # 由于气管的存在会导致左右肺，两个X轴区间有交叉，故只根据区间长度差超过3倍/全包裹，或宽度相加超过边长则pass该连通域
                if float(x_first_length) / x_second_length > 3.0 or float(x_second_length) / x_first_length > 3.0 or \
                        x_first_length + x_second_length > labels.shape[2] or (x_second_start > x_first_start and x_second_end < x_first_end):
                    if i == len(order) - 1:
                        ret_mask = labels == vals[order[0]]
                        return ret_mask.astype('uint8'), 1
                    continue

                ret_mask = np.logical_or((labels == vals[order[0]]), (labels == vals[order[i]])).astype('uint8')
                return ret_mask, 2
    else:
        return mask, 0


def postprocess_mask(mask, iterations=5):
    """
    :brief 后处理分割得到的mask
    :param[in] mask: 3d array, 肺mask，shape: [z, y, x]
    :param[in] iterations: int, 膨胀次数
    :return: 后处理后的mask
    """
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(mask, structure=struct, iterations=iterations)
    return dilatedMask


def fill_2d_hole(mask):
    """
    :brief 在3D图像上逐层处理2D图像填充mask内的小孔
    :param[in] mask: 3d array, 二值化mask, shape: [z, y, x]
    :return: 填充了小孔洞的二值化mask, uint8
    """
    for i in range(mask.shape[0]):
        label = measure.label(mask[i])
        properties = measure.regionprops(label)
        for prop in properties:
            bb = prop.bbox
            mask[i][bb[0]:bb[2], bb[1]:bb[3]] = mask[i][bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image

    return mask.astype('uint8')


def find_chest(mask, is_local=False):
    """
    :brief 提取胸部区域
    :param[in&out] mask: 3d array, 经过卡阈值等简单操作得到的粗糙mask
    :return: 得到胸部区域的mask, bool
    """
    # 开操作一次去除背板线的连接，防止处理误将全肺判为局部肺封边时造成连通域，部分case背板较粗，多次迭代可将其断开
    if is_local:
        struct = generate_binary_structure(3, 1)
        struct[0, :] = 0  # Z轴不进行开操作
        struct[2, :] = 0  # Z轴不进行开操作
        mask = binary_opening(mask, structure=struct, iterations=2)
        init_mask = mask.copy()
        # 上面主要是在去背板的线连接，去除背板后这里再加入封边处理局部肺，应无影响
        # 全封死以处理极限情况（几乎无胸腔）的局部肺
        mask[:, 0:1, :] = True
        mask[:, -1:, :] = True
        mask[:, :, 0:1] = True
        mask[:, :, -1:] = True

        # mask[:, 0:1, 2:-2] = True
        # mask[:, -1:, 2:-2] = True
        # mask[:, 2:-2, 0:1] = True
        # mask[:, 2:-2, -1:] = True

    for iz in range(mask.shape[0]):
        mask[iz] = ndi.binary_fill_holes(mask[iz])  # fill body
        label_image = measure.label(mask[iz])
        regions_image = measure.regionprops(label_image)
        max_area = 0
        seq = 0
        for region in regions_image:
            if region.area > max_area:
                max_area = region.area
                seq = region.label
        # 这里两种同等效果赋值方式的速度好像有挺大差异，待测
        # mask[iz] = label_image == seq
        mask[iz] = np.in1d(label_image, [seq]).reshape(label_image.shape)

    return mask


# def save_png(mask,path_id):
#     if not os.path.exists('D:\work\lung_mask_test\png{}\\'.format(path_id)):
#         os.mkdir('D:\work\lung_mask_test\png{}\\'.format(path_id))
#     for i in range(len(mask)):
#         cv2.imwrite('D:\work\lung_mask_test\png{}\\'.format(path_id) + str(i) + ".jpg",  imutils.resize(mask[i], height=151,width=240)*255)

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


def segment_lung_mask(dcm_img, chest_th=-500, lung_th=-320, min_object_size=10, max_areas=250000, min_areas=2000):
    """
    :brief 从原始dcm序列中完成初步的肺部区域图像分割和处理
    :param[in] dcm_img: 3d array, 原始肺部CT序列, shape: [z, y, x]
    :param[in] chest_th: int, 胸部阈值, 默认为-300
    :param[in] lung_th: int, 肺部阈值，默认为-320
    :param[in] min_object_size: int, 最小有效物体体积，默认为300
    :param[in] max_areas: int, 最大区域面积，默认为250000
    :param[in] min_areas: 最小区域面积，默认为4000
    :return: 初步分割得到的肺部二值化mask，3d array, uint8, shape: [z, y, x]
    """

    mask = dcm_img > chest_th # -500

    mask = find_chest(mask)

    temp_x = mask * dcm_img # 粗糙肺mask

    fine_mask = temp_x < lung_th # -300

    fine_mask = ndimage.median_filter(fine_mask.astype('uint8'), size=(1, 3, 3))

    fine_mask, num_regions = largest_one_or_two_label_volume(fine_mask, temp_x < lung_th, bg=0, )

    fine_mask = fill_2d_hole(fine_mask)
    ret_mask = fine_mask



    return ret_mask

if __name__ == '__main__':
    data_path = r'D:\work\dataSet\airway\train'
    output_path = r'D:\work\dataSet\airway\lung_mask'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in [i for i in os.listdir(data_path) if not i.endswith('_label.nii.gz')]:
        print(i)
        data_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path,i)))


        print(data_arr.shape)
        mask_arr = segment_lung_mask(data_arr)
        box = compute_lung_extendbox(mask_arr)
        crop_data = data_arr[box[0, 0]: box[0, 1], box[1, 0]: box[1, 1], box[2, 0]: box[2, 1]]
        print(crop_data.shape)
        img_itk = sitk.GetImageFromArray(mask_arr)

        img_itk.SetOrigin(sitk.ReadImage(os.path.join(data_path,i)).GetOrigin())
        img_itk.SetSpacing(sitk.ReadImage(os.path.join(data_path,i)).GetSpacing())

        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(output_path,i.replace('.nii.gz','_lungmask.mhd')))
        writer.Execute(img_itk)
        print('Success write to {}'.format(os.path.join(output_path,i.replace('.nii.gz','_lungmask.mhd'))))