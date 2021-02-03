# coding:utf-8
"""
@time: 19-2-22
@author: zol
@contact: sbzol.chen@gmail.com
@file:  preprocess_25d
@desc: None
@==============================@
@      _______ ____       __   @
@     / __/ _ )_  / ___  / /   @
@    _\ \/ _  |/ /_/ _ \/ /    @
@   /___/____//___/\___/_/     @
@                        SBZol @ 
@==============================@
"""
import os
import numpy as np
import nibabel as nib
from nilearn import image

from preprocess import PreProcess
from utils import split_filename, reslice_nii, get_main_axis


class PreProcess25D(PreProcess):
    def __init__(self):
        super(PreProcess25D, self).__init__()

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