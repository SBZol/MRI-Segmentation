# _*_ coding:utf-8 _*_
"""
@time: 2018/6/8
@author: cliff
@contact: zshtom@163.com
@file: utils.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import os
from glob import glob
import re
import random
import numpy as np
import time

import nibabel as nib
import dicom2nifti
import json
from keras.utils import to_categorical
from dipy.align.reslice import reslice

from dataIO import compress_gz


def random_sample(orin_list, sample_num):
    """
    对数组随机抽样, 不改变原有的数组, 返回抽样得到的数组和抽样得到数组和原数组的差集

    :param orin_list: 原数组
    :param sample_num: 抽样数量
    :return: sample_set(抽样得到的数组) difference_set(抽样得到数组和原数组的差集）
    """
    # 随机抽样
    sample_set = random.sample(orin_list, sample_num)

    # 计算抽样得到数组和原数组的差集
    difference_set = list(set(orin_list).difference(set(sample_set)))
    return sample_set, difference_set


def read_nii_vol(nii_path):
    """
    读取nii数据的图像矩阵.

    Arguments:
        nii_path -- nii 数据的路径

    Returns:
        nii_data -- 读取到的图像矩阵

    """
    nii = nib.load(nii_path)
    nii_data = nii.get_data()
    return nii_data


def np_2_nii(original_nii, nii_npy, save_path='', need_save=False, header=None):
    """
    将numpy array 转换成 nii 并在 save_path 路径下保存为.nii文件.

    Arguments:
        original_nii -- 原始的 nii, 用于提取 nii 图像信息
        nii_npy -- 用于转换成 nii 的 numpy array
        save_path -- 最终生成 nii 文件的保存路径
        header -- nii的huader 没有的话默认和 originnal_nii 一致

    """

    # 判断 original_nii 的输入类型是否是 string (nii 文件的路径)
    is_string = isinstance(original_nii, str)

    # 若输入的 original_nii 是 string 类型, 则通过 nib.load() 获取 nii 数据
    original_img = nib.load(original_nii) if is_string else original_nii
    nii_header = original_img.header if (header is None) else header

    # print('------> new nii dim:' + str(nii_header['dim']))
    # print('------> nwe nii pixdim:' + str(nii_header['pixdim']))

    # 调用 nib.Nifti1Image() 将 numpy 转换成 nii.
    img = nib.Nifti1Image(nii_npy, affine=original_img.affine, header=nii_header)

    # 保存 nii 文件到 save_path 下
    if need_save:
        nib.save(img, save_path)

    return img


def split_filename(path):
    """
    分割路径并得到路径中的文件名.

    Arguments:
        path -- 需要分割的路径

    Returns:
        filename -- 分割后得到的文件名
    """
    filename = os.path.split(path)[-1]
    return filename


def one_hot_matrix(labels, c):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.
    Arguments:
        labels -- vector containing the labels
        c -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """

    # 计算 one_hot

    one_hot = to_categorical(labels, num_classes=c)

    return one_hot


def dicom2nii(dicom_path, save_path):
    """
    将 Dicom 文件夹下的文件转换成一个 nii 文件,并保存在当前目录下
    Arguments:
        dicom_path -- dicom 文件夹路径
        save_path -- 转换后保存的路径

    Returns:
    nii_path -- 生成的 nii 文件路径
    """
    print('------> dicom files:' + str(len(os.listdir(dicom_path))))
    dicom2nifti.convert_directory(dicom_path, save_path)

    all_files = glob(os.path.join(save_path, "*"), recursive=False)
    nii_path = None

    for file in all_files:
        if re.search('.nii.gz', split_filename(file)):
            nii_path = file
    print('------> output nii path: ' + nii_path)
    return nii_path


def compress_nii(input_path, output_path, remove_origin=False):
    """
    压缩nii文件
    Arguments:
        input_path -- nii文件路径
        output_path -- 压缩输出路径
        remove_origin -- 压缩后是否删除原文件,默认False
    Returns:
    nii_path -- 生成的 nii 文件路径
    """
    print('------> compress nii file: ' + output_path)
    compress_gz(input_path, output_path)
    if remove_origin:
        os.remove(input_path)


def to_json(data_array, uid_array, file_path):
    """
    json转换
    Arguments:
        data_array -- 数据数组
        uid_array -- 数据对应的key值
        file_path -- 存储数据
    """
    jsondata = {}
    for i in range(len(data_array)):
        jsondata[uid_array[i]] = data_array[i]

    with open(file_path, "w") as file:
        json.dump(jsondata, file)


def calculate_label_weight(dir_path, num_classes):
    """
    计算文件夹下所有label各类面积平均值的倒数作为权重
    Arguments:

    """
    listdir = os.listdir(dir_path)
    weights = np.zeros((num_classes))
    for nii_path in listdir:
        nii_data = nib.load(nii_path).get_data()
        for i in num_classes:
            area = np.sum(nii_data == i)
            weights[i] = weights[i] + area
    # 求均值
    weights /= len(listdir)

    # 每个元素求倒
    weights = 1. / weights
    return weights


def get_main_axis(data):
    """
    获取3D图像的主轴,以维度最小的主
    Arguments:
        data: 输入的data

    Returns:
        main_voxel: 主轴, 0,1,2 分别代表 x,y,z
    """
    shape = data.shape
    main_axis = np.argmin(shape)
    return main_axis


def reslice_nii(nii, target_zooms=(1., 1., 1.), key='input'):
    """
    统一 nii 的 spacing
    Arguments:
        :param nii: nii 数据
        :param target_zooms： 目标spacing
        :param key： 'input' or 'label' 决定了插值的方法
    Returns:
        :return data: reslice 后的数据
        :return affine：新的affine
    """
    affine = nii.affine
    zooms = nii.header.get_zooms()[:3]

    if key == 'input':
        data, affine = reslice(nii.get_data(), affine, zooms, target_zooms)
    elif key == 'label':
        print('************')
        print(np.unique(nii.get_data()))

        data, affine = reslice(nii.get_data(), affine, zooms,
                               target_zooms, mode='nearest', order=0)
        print(np.unique(data))
        print('************')
    return data, affine


class TimeCalculation(object):
    """
    用于统计时间的工具类,调用begin后再调用end就会统计出时间并输出
    Arguments:
        name: 名称,统计完成后输出用的字段(可选)
    """

    def __init__(self, name='time'):
        self.name = name
        self.starTime = 0
        self.endTime = 0

    def begin(self):
        self.starTime = time.time()

    def end(self):
        self.endTime = time.time()
        print('~~~~~~~~~~~~~~~~~~~> timeCalculation: ' + self.name + ' = ' + str(self.endTime - self.starTime))


if __name__ == '__main__':
    path = '/media/chenzhuo/50FA53DAFA53BB44/DL/Data/fcon_1000_anat/AnnArbor_a'
    for sub_dir in os.listdir(path):
        print(sub_dir)
    pass
