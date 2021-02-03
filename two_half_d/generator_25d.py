# coding:utf-8
"""
@time: 19-2-25
@author: zol
@contact: sbzol.chen@gmail.com
@file:  generator_25d
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
from keras.utils import Sequence
import random
import cv2

from utils import one_hot_matrix


class Generator4Train25D(Sequence):

    def __init__(self, data_path, batch_size=4, shuffle=True, n_classes=16, normalize_func=None):
        """
        构造函数

        :param data_path: 保存image, label的路径
        :param batch_size: batch_size
        :param shuffle: 随机打乱
        :param n_classes: 分类数
        :param normalize_func 用于归一化数据的函数, 参数是一个 numpy array. 例如 normalization(data)
        """
        self.data_path = data_path

        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize_func = normalize_func

        self.data_list = os.listdir(data_path)
        self.data_list.sort()

        self.indexes = np.arange(len(self.data_list))
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of batch
        """
        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        axial_img_batch = None
        sagittal_img_batch = None
        coronal_img_batch = None
        axial_label_batch = None
        sagittal_label_batch = None
        coronal_label_batch = None
        for i in indexes:
            img_file_name = self.data_list[i]
            data = np.load(os.path.join(self.data_path, img_file_name))

            image = data[0]
            label = data[1]

            axial_img = image[0][np.newaxis, :, :]
            sagittal_img = image[1][np.newaxis, :, :]
            coronal_img = image[2][np.newaxis, :, :]

            axial_label = label[0][np.newaxis, :, :]
            sagittal_label = label[1][np.newaxis, :, :]
            coronal_label = label[2][np.newaxis, :, :]

            if self.normalize_func:
                axial_img = self.normalize_func(axial_img)
                sagittal_img = self.normalize_func(sagittal_img)
                coronal_img = self.normalize_func(coronal_img)
            # print(axial_img.shape)
            # print(sagittal_img.shape)
            # print(coronal_img.shape)
            # print('_________')

            axial_img_batch = axial_img if (axial_img_batch is None) else np.concatenate((axial_img_batch, axial_img),
                                                                                         axis=0)
            sagittal_img_batch = sagittal_img if (sagittal_img_batch is None) else np.concatenate(
                (sagittal_img_batch, sagittal_img),
                axis=0)
            coronal_img_batch = coronal_img if (coronal_img_batch is None) else np.concatenate(
                (coronal_img_batch, coronal_img),
                axis=0)

            axial_label_batch = axial_label if (axial_label_batch is None) else np.concatenate(
                (axial_label_batch, axial_label),
                axis=0)
            sagittal_label_batch = sagittal_label if (sagittal_label_batch is None) else np.concatenate(
                (sagittal_label_batch, sagittal_label),
                axis=0)
            coronal_label_batch = coronal_label if (coronal_label_batch is None) else np.concatenate(
                (coronal_label_batch, coronal_label),
                axis=0)

        if self.n_classes == 2:
            axial_label_batch[axial_label_batch != 0] = 1
            axial_label_batch = one_hot_matrix(axial_label_batch, self.n_classes)

            sagittal_label_batch[sagittal_label_batch != 0] = 1
            sagittal_label_batch = one_hot_matrix(sagittal_label_batch, self.n_classes)

            coronal_label_batch[coronal_label_batch != 0] = 1
            coronal_label_batch = one_hot_matrix(coronal_label_batch, self.n_classes)
        else:
            axial_label_batch = one_hot_matrix(coronal_label_batch, self.n_classes)
            sagittal_label_batch = one_hot_matrix(coronal_label_batch, self.n_classes)
            coronal_label_batch = one_hot_matrix(coronal_label_batch, self.n_classes)

        axial_img_batch = axial_img_batch[..., np.newaxis]
        sagittal_img_batch = sagittal_img_batch[..., np.newaxis]
        coronal_img_batch = coronal_img_batch[..., np.newaxis]

        return [axial_img_batch, sagittal_img_batch, coronal_img_batch], sagittal_label_batch

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':
    from pathvariable import *
    import matplotlib.pyplot as plt

    gen = Generator4Train25D(data_path=TRAIN_DATA_PATH, batch_size=3,
                             shuffle=True, n_classes=4)

    # gen = Generator4Train3D(image_path=TRAIN_IMAGE_PATH, label_path=TRAIN_LABEL_PATH,
    #                         batch_size=4,
    #                         shuffle=True, n_classes=2, normalize_func=None)

    for index in range(gen.__len__()):
        _ = gen.__getitem__(index)
