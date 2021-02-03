# _*_ coding:utf-8 _*_
"""
@time: 2018/6/8
@author: cliff
@contact: zshtom@163.com
@file: preprocess.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import os
import numpy as np
import nibabel as nib
from nilearn import image

from utils import split_filename, reslice_nii, get_main_axis


class PreProcess(object):

    @classmethod
    def normalize(cls, vol):
        max_value = np.max(vol)
        min_value = np.min(vol)
        nor_vol = (vol - min_value) / (max_value - min_value)
        return nor_vol

    @classmethod
    def standardiztion(cls, vol):
        mean = np.mean(vol)
        std = np.std(vol)
        nor_vol = (vol - mean) / std
        return nor_vol

    @classmethod
    def padding(cls, data, target_shape, log=True):
        main_axis = get_main_axis(data)
        # 变换主轴
        transpose_shape_dict = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}

        r_nii_data = np.transpose(data, transpose_shape_dict[main_axis])

        if log:
            print('------> transpose shape: ' + str(r_nii_data.shape))

        # padding
        x_padding = (0, 0) if r_nii_data.shape[0] >= 32 else (0, 32 - r_nii_data.shape[0])
        y_padding = (0, (target_shape[1] - r_nii_data.shape[1]) if (target_shape[1] - r_nii_data.shape[1]) >= 0 else 0)
        z_padding = (0, (target_shape[2] - r_nii_data.shape[2]) if (target_shape[2] - r_nii_data.shape[2]) >= 0 else 0)

        if log:
            print('------> padding:')
            print('------> x_padding:' + str(x_padding))
            print('------> y_padding:' + str(y_padding))
            print('------> z_padding:' + str(z_padding))

        padding_data = np.pad(r_nii_data, (x_padding, y_padding, z_padding), mode='constant')
        return padding_data


class PreProcess4SSpacing(PreProcess):
    def __init__(self):
        super(PreProcess4SSpacing, self).__init__()

    @classmethod
    def preprocess(cls, origin_nii, standard_shape, key='input', normalize=None):

        print('------> reslice zooms: (1., 1., 1.)')
        reslice_data, reslice_affine = reslice_nii(origin_nii, key=key)

        reslice_data_shape = reslice_data.shape
        main_axis = get_main_axis(reslice_data)
        print('------> reslice data shape: ' + str(reslice_data_shape))

        print('------> main axis:' + str(main_axis))

        # 变换主轴
        transpose_shape_dict = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}

        r_nii_data = np.transpose(reslice_data, transpose_shape_dict[main_axis])

        print('------> transpose shape: ' + str(r_nii_data.shape))

        # 这里需要针对 reslice 后的 shape 大于 input_shape 的情况进行裁剪,目前没做处理
        if r_nii_data.shape[1] > standard_shape[1] or r_nii_data.shape[2] > standard_shape[1]:
            print('xxxxxxxxxxx> error shape: ' + str(r_nii_data.shape))
            return

        # padding
        x_padding = (0, 0) if r_nii_data.shape[0] >= 32 else (0, 32 - r_nii_data.shape[0])
        y_padding = (0, standard_shape[1] - r_nii_data.shape[1])
        z_padding = (0, standard_shape[2] - r_nii_data.shape[2])
        print('------> padding:')
        print('------> x_padding:' + str(x_padding))
        print('------> y_padding:' + str(y_padding))
        print('------> z_padding:' + str(z_padding))
        r_nii_data = np.pad(r_nii_data, (x_padding, y_padding, z_padding), mode='constant')
        print('------> padding shape:' + str(r_nii_data.shape))

        if normalize == 'normalize':
            r_nii_data = cls.normalize(r_nii_data)
            print('------> normalize data')

        elif normalize == 'standardiztion':
            r_nii_data = cls.standardiztion(r_nii_data)
            print('------> standardiztion data')

        return r_nii_data, reslice_data, reslice_affine, main_axis, [x_padding, y_padding, z_padding]


class PreProcess4MSpacing(PreProcess):
    # 用于mutiple spacing, 还不成熟，暂时禁用

    def __init__(self):
        super(PreProcess4MSpacing, self).__init__()

    @classmethod
    def preprocess(cls, origin_nii, standard_shape, normalize=None):

        origin_nii_data = origin_nii.get_data()
        main_axis = get_main_axis(origin_nii_data)
        origin_nii_data_shape = origin_nii_data.shape
        print('------> reslice data shape: ' + str(origin_nii_data_shape))

        print('------> main axis:' + str(main_axis))

        # 变换主轴
        transpose_shape_dict = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}

        r_nii_data = np.transpose(origin_nii_data, transpose_shape_dict[main_axis])

        print('------> transpose shape: ' + str(r_nii_data.shape))

        # padding
        x_padding = (0, 0) if r_nii_data.shape[0] >= 32 else (0, 32 - r_nii_data.shape[0])
        y_padding = (0, (standard_shape[1] - r_nii_data.shape[1]) if (standard_shape[1] - r_nii_data.shape[1]) > 0 else 0)
        z_padding = (0, (standard_shape[2] - r_nii_data.shape[2]) if (standard_shape[2] - r_nii_data.shape[2]) > 0 else 0)
        print('------> padding:')
        print('------> x_padding:' + str(x_padding))
        print('------> y_padding:' + str(y_padding))
        print('------> z_padding:' + str(z_padding))
        r_nii_data = np.pad(r_nii_data, (x_padding, y_padding, z_padding), mode='constant')
        print('------> padding shape:' + str(r_nii_data.shape))

        if normalize == 'normalize':
            r_nii_data = cls.normalize(r_nii_data)
            print('------> normalize data')

        elif normalize == 'standardiztion':
            r_nii_data = cls.standardiztion(r_nii_data)
            print('------> standardiztion data')

        return r_nii_data, main_axis, [x_padding, y_padding, z_padding]

    @classmethod
    def create_train_data(cls, train_data_paths, output_path, standard_shape):
        """
        对原始的 inputs，labels 进行 resample, resize 后保存到本地的路径.

        Arguments:
            train_data_paths -- 存放原始 input 路径数组, 可通过 utils.py 中函数 get_original_data_paths() 获取
            output_path -- 新的 label 存放的路径
            standard_shape -- 标准的 shape(即网络输入的shpae), reslice 后会把结果 padding or clips 到标准的 shape
        """

        def pre_data(nii_path, output_path, key='input'):
            """
            对单个 nii 数据 preprocess 后并在 save_path 路径下保存为.nii文件

            :param nii_path: 需要 reslice 的 nii 文件路径
            :param output_path: reslice 后新的 nii 文件输出路径
            :param key： 'input' or 'label' 决定了插值的方法
            """
            nii = image.load_img(nii_path)
            transpese_data, main_axis, padding = cls.preprocess(nii, standard_shape)
            # 一些奇怪的label数据得到的不是整数,如输出值是2,确变成了1.999999,这里需要四舍五入
            if key == 'label':
                transpese_data = np.rint(transpese_data)

            transpese_nii = nib.Nifti1Image(transpese_data, affine=nii.affine, header=nii.header)
            nib.save(transpese_nii, output_path)

        for data_paths in train_data_paths:
            img = data_paths['image']
            label = data_paths['label']
            dir_name = label.split(os.sep)[-2]
            dir_path = os.path.join(output_path, dir_name)

            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                print('create dir:' + dir_path)

            img_ouput_path = os.path.join(dir_path, split_filename(img))
            label_ouput_path = os.path.join(dir_path, split_filename(label))

            pre_data(img, img_ouput_path)
            pre_data(label, label_ouput_path, key='label')

            print(img_ouput_path)
            print(label_ouput_path)
            print('-------------------------------------------------------------->')
        print('----------->reslice done!')

    @classmethod
    def diversification_spacing(cls, train_data_paths, output_path, ratio=2.0):

        def enhance_spacing(nii_path, output_path, key='input'):
            nii = image.load_img(nii_path)
            target_sapcing = (
            nii.header['pixdim'][1] / ratio, nii.header['pixdim'][2] / ratio, nii.header['pixdim'][3] / ratio)
            reslice_data, reslice_affine = reslice_nii(nii, target_zooms=target_sapcing, key=key)
            # 一些奇怪的label数据得到的不是整数,如输出值是2,确变成了1.999999,这里需要四舍五入
            if key == 'label':
                reslice_data = np.rint(reslice_data)

            transpese_nii = nib.Nifti1Image(reslice_data, affine=reslice_affine, header=nii.header)
            nib.save(transpese_nii, output_path)

        for data_paths in train_data_paths:
            img = data_paths['image']
            label = data_paths['label']
            dir_name = label.split(os.sep)[-2]
            dir_name = dir_name + '_' + str(ratio)
            dir_path = os.path.join(output_path, dir_name)

            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                print('create dir:' + dir_path)

            img_ouput_path = os.path.join(dir_path, split_filename(img))
            label_ouput_path = os.path.join(dir_path, split_filename(label))

            enhance_spacing(img, img_ouput_path)
            enhance_spacing(label, label_ouput_path, key='label')

            print(img_ouput_path)
            print(label_ouput_path)
            print('-------------------------------------------------------------->')
        print('----------->reslice done!')


if __name__ == '__main__':
    import nilearn.image as image

    nii = nib.load('/media/chenzhuo/50FA53DAFA53BB44/DL/Data/Beijing_Zang/part1_sub08001/2018_1_8001mprage.nii')
    header = nii.header
    x = header['pixdim'][1]
    y = header['pixdim'][2]
    z = header['pixdim'][3]
    print(type(x))
    pass
