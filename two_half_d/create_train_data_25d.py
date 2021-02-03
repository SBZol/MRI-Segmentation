# coding:utf-8
"""
@time: 19-2-22
@author: zol
@contact: sbzol.chen@gmail.com
@file:  create_train_data_25d
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
import math
import numpy as np
import nibabel as nib
from nilearn import image

from create_train_data import CreateTrainData
from preprocess_25d import PreProcess25D
from utils import split_filename, one_hot_matrix, random_sample


class CreateTrainData25D(object):
    def __init__(self,
                 train_data_paths,
                 output_path,
                 nii_ouput_path,
                 target_shape=(32, 256, 256),
                 two_half_shape=(32, 32),
                 num_classes=4):
        """
        构造方法

        :param train_data_paths: 存放原始 input 路径数组, 可通过 utils.py 中函数 get_original_data_paths() 获取
        :param output_path: 处理后的图像和label存放的路径
        :param nii_ouput_path: relice后的label image 的nii格式存放路径
        :param target_shape: 标准的 shape(即网络输入的shpae), reslice 后会把结果 padding or clips 到标准的 shape
        :param two_half_shape: 2.5D 块每个面的大小, 三个面shape一致
        :param num_classes: 分类数量
        """
        self.train_data_paths = train_data_paths
        self.output_path = output_path
        self.nii_ouput_path = nii_ouput_path
        self.target_shape = target_shape
        self.two_half_shape = two_half_shape
        self.num_classes = num_classes

    def create_train_data(self):
        """
        对原始的 inputs，labels 进行 resample, resize 后保存到本地的路径.
        """

        for data_paths in self.train_data_paths:
            img = data_paths['image']
            label = data_paths['label']
            dir_name = label.split(os.sep)[-2]
            dir_path = os.path.join(self.nii_ouput_path, dir_name)

            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                print('create dir:' + dir_path)

            img_ouput_path = os.path.join(dir_path, split_filename(img))
            label_ouput_path = os.path.join(dir_path, split_filename(label))

            # 对 img 和 label 进行训练的预处理
            img_data = self.process_image(img, img_ouput_path)
            label_data = self.process_image(label, label_ouput_path, key='label')

            # 对 img 和 label 进行切块并保存为 npy 文件
            ouput_path = os.path.join(self.output_path, dir_name)
            self.slice_and_save_data(img_data, label_data, ouput_path)

        print('-----------> Create train Data done! (SS)')

    def process_image(self, nii_path, output_path, key='input'):
        """
        对单个 nii 数据 reslice 后并在 save_path 路径下保存为.nii文件

        :param nii_path: 需要 reslice 的 nii 文件路径
        :param output_path: 处理后的 data 保存为 nii 的路径
        :param key： 'input' or 'label' 决定了插值的方法,详情见 preprocess 函数的注释
        :return: 经过训练预处理后的 data
        """
        nii = image.load_img(nii_path)
        transpose_data, reslice_data, reslice_affine, _, _ = PreProcess25D.preprocess(nii,
                                                                                      self.target_shape,
                                                                                      key)
        # 一些奇怪的label数据得到的不是整数,如输出值是2,确变成了1.999999,这里需要四舍五入
        if key == 'label':
            transpose_data = np.rint(transpose_data)

        # 保存为nii
        transpese_nii = nib.Nifti1Image(transpose_data, affine=reslice_affine, header=nii.header)
        nib.save(transpese_nii, output_path)
        print('save nii: ' + output_path)

        return transpose_data

    def slice_and_save_data(self, imgs, labels, output_dir_path):
        """
        对 img 和 label 按 x 2.5d切块

        :param imgs: 需要切块的 image
        :param labels: image 对应的 label
        :param output_dir_path: 切块保存的根目录
        """
        print(imgs.shape)
        print(labels.shape)
        print('................')

        image_shape = imgs.shape
        step = self.two_half_shape[0]  # 裁剪步伐
        half_step = int(self.two_half_shape[0] / 2)  # 裁剪起始位置

        # 对图像x轴上下padding 16 层, 解决裁剪2.5D块时x轴上出现的边缘问题
        x_padding = (half_step, half_step)
        y_padding = (0, 0)
        z_padding = (0, 0)
        print('------> padding:')
        print('------> x_padding:' + str(x_padding))
        print('------> y_padding:' + str(y_padding))
        print('------> z_padding:' + str(z_padding))
        padding_image = np.pad(imgs, (x_padding, y_padding, z_padding), mode='constant')
        padding_labels = np.pad(labels, (x_padding, y_padding, z_padding), mode='constant')
        print('------> padding shape:' + str(padding_image.shape))

        for x in range(0, image_shape[0]):
            for y in range(half_step, image_shape[1], step):
                for z in range(half_step, image_shape[2], step):
                    sagittal_slice_img, coronal_slice_img, axial_slice_img = self.get_25d_slice(padding_image,
                                                                                                x, y, z,
                                                                                                half_step,
                                                                                                self.two_half_shape)
                    sagittal_slice_lb, coronal_slice_lb, axial_slice_lb = self.get_25d_slice(padding_labels,
                                                                                             x, y, z,
                                                                                             half_step,
                                                                                             self.two_half_shape)
                    two_half_block = [[sagittal_slice_img, coronal_slice_img, axial_slice_img],
                                      [sagittal_slice_lb, coronal_slice_lb, axial_slice_lb],
                                      ]
                    two_half_block = np.array(two_half_block)

                    save_path = output_dir_path + '_25d_slice_' + str(x) + '_' + str(y) + '_' + str(z) + '.npy'
                    np.save(save_path, two_half_block)
                    print('save_path:', save_path)
                    print('--------------')

    @classmethod
    def get_25d_slice(cls, data, x, y, z, half_step, target_shape):
        sagittal_slice = data[
                         half_step + x,
                         y - half_step:y + half_step,
                         z - half_step:z + half_step
                         ]

        coronal_slice = data[
                        x:x + 2 * half_step,
                        y,
                        z - half_step:z + half_step]

        axial_slice = data[
                      x:x + 2 * half_step,
                      y - half_step:y + half_step,
                      z]

        # print('x:', half_step + x, ' y:', y - half_step, y + half_step, ' z:', z - half_step, z + half_step)
        # print('x:', x, x + 2 * half_step, ' y:', y, 'z:', z - half_step, z + half_step)
        # print('x:', x, x + 2 * half_step, 'y', y - half_step, y + half_step, ' z:', z)

        assert coronal_slice.shape == axial_slice.shape == sagittal_slice.shape == target_shape, "output shape error"

        return sagittal_slice, coronal_slice, axial_slice


if __name__ == '__main__':
    from pathvariable import *

    # 检测文件夹路径是否存在，不存在则创建
    check_path()

    # step1：生成 train set
    train_data_path = CreateTrainData.get_original_data_paths(ORIGINAL_PATH, 'Brain_4_labels.nii')
    CreateTrainData25D(train_data_paths=train_data_path,
                       output_path=TRAIN_DATA_PATH,
                       nii_ouput_path=TRAIN_DATA_NII_PATH,
                       target_shape=(32, 256, 256),
                       two_half_shape=(32, 32),
                       num_classes=4).create_train_data()
    print(train_data_path)
