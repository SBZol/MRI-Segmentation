"""
@time: 18-9-13
@author: zol
@contact: 13012215283@sina.cn
@file:  create_train_data
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import os
import re
import math
import random
import numpy as np
import nibabel as nib
from nilearn import image
from glob import glob

from utils import split_filename, one_hot_matrix, random_sample
from preprocess import PreProcess4SSpacing, PreProcess4MSpacing


class CreateTrainData(object):
    def __init__(self, train_data_paths, output_path, nii_ouput_path,target_shape=(32, 256, 256), step=16, num_classes=4):
        """
        构造方法

        :param train_data_paths: 存放原始 input 路径数组, 可通过 utils.py 中函数 get_original_data_paths() 获取
        :param output_path: 新的 label 存放的路径
        :param nii_ouput_path: relice后的label image 的nii格式存放路径
        :param target_shape: 标准的 shape(即网络输入的shpae), reslice 后会把结果 padding or clips 到标准的 shape
        :param step: 在 SS (Static Spacing) 的训练方法会用到, 即是在主轴上切块的时候, 间隔 step 取一个切块, 默认值为16
        :param num_classes: 分类数量
        """
        self.train_data_paths = train_data_paths
        self.output_path = output_path
        self.nii_ouput_path = nii_ouput_path
        self.target_shape = target_shape
        self.step = step
        self.num_classes = num_classes

    def create_train_data(self):
        """
        创建训练用的 data 的函数, 在子类中重写该函数实现不同的方法
        """
        pass

    @classmethod
    def get_original_data_paths(cls, original_path, label_search_name):
        """
        从原始数据中获取所有 inputs 和 labels 的路径，返回为array.

        Arguments:
            original_path -- 原始数据目录的路径
            label_search_name -- 路径下查到 label 的关键字

        Returns:
            train_data_paths -- 存放所有image 和 label路径的数组,数组的每个元素对应一个字典 {image : ... , label :...}
        """

        # 获取 original_path 下的所有文件夹路径
        all_paths = glob(os.path.join(original_path, "*"), recursive=False)

        train_data_paths = []

        # 遍历每个文件夹, 搜索需要的 input 和 label ,并把查到的路径存放到相对应的路径数组中
        for sub_path in all_paths:
            all_files = glob(os.path.join(sub_path, "*"), recursive=False)
            tmp_dict = {}
            for file in all_files:
                if split_filename(file) == 'mprage_anonymized.nii':
                    tmp_dict['image'] = file
                if re.search(label_search_name, split_filename(file)):
                    tmp_dict['label'] = file
            train_data_paths.append(tmp_dict)

        return train_data_paths


class CreateTrainData4SS(CreateTrainData):

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
        transpose_data, reslice_data, reslice_affine, _, _ = PreProcess4SSpacing.preprocess(nii,
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

    def slice_and_save_data(self, imgs, labels, output_path):
        """
        对 img 和 label 按 x 轴为主轴按照 self.slice_size 和 self.step 切块

        :param imgs: 需要切块的 image
        :param labels: image 对应的 label
        :param output_path: 切块保存的根目录
        """

        voxel_x = imgs.shape[0]
        target_gen_size = self.target_shape[0]

        # 计算切块需要循环的次数
        range_val = int(math.ceil((voxel_x - target_gen_size) / self.step) + 1)

        for i in range(range_val):
            start_num = i * self.step
            end_num = start_num + target_gen_size

            if end_num <= voxel_x:
                # 数据块长度没有超出x轴的范围,正常取块
                slice_img = imgs[start_num:end_num, :, :]
                slice_label = labels[start_num:end_num, :, :]
            else:
                # 数据块长度超出x轴的范围, 从最后一层往前取一个 batch_gen_size 大小的块作为本次获取的数据块
                slice_img = imgs[(voxel_x - target_gen_size):voxel_x, :, :]
                slice_label = labels[(voxel_x - target_gen_size):voxel_x, :, :]

            # slice_img = slice_img[:, :, :]
            # one_hot_label = one_hot_matrix(slice_label, self.num_classes)

            # 保存切块
            save_path = output_path + '_slice_' + str(i) + 'image_slice.npy'

            data_array = [slice_img, slice_label]
            data_array = np.array(data_array)

            np.save(save_path, data_array)

            print('save data slice: ' + save_path)
            print('------')
        print('------------------------------------------------------------')
