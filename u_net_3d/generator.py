# _*_ coding:utf-8 _*_
"""
@time: 2018/6/22
@author: zol
@contact: 13012215283@sina.cn
@file: generator.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import os
import numpy as np
from keras.utils import Sequence
import random
import cv2

from utils import one_hot_matrix


class Generator4Train3D(Sequence):

    def __init__(self, data_path, batch_size=4, shuffle=True, n_classes=4, normalize_func=None):
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
        img_batch = None
        label_batch = None

        for i in indexes:
            img_file_name = self.data_list[i]
            data = np.load(os.path.join(self.data_path, img_file_name))

            image = data[0][np.newaxis,...]
            label = data[1][np.newaxis,...]

            if self.normalize_func:
                image = self.normalize_func(image)

            img_batch = image if (img_batch is None) else np.concatenate((img_batch, image),
                                                                         axis=0)

            label_batch = label if (label_batch is None) else np.concatenate(
                (label_batch, label),
                axis=0)

        if self.n_classes == 2:
            label_batch[label_batch != 0] = 1
            label_batch = one_hot_matrix(label_batch, self.n_classes)

        else:
            label_batch = one_hot_matrix(label_batch, self.n_classes)

        img_batch = img_batch[..., np.newaxis]

        return img_batch, label_batch

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':
    from pathvariable import *
    import matplotlib.pyplot as plt

    gen = Generator4Train3D(data_path=TRAIN_DATA_PATH, batch_size=2,
                            shuffle=True, n_classes=4)

    # gen = Generator4Train3D(image_path=TRAIN_IMAGE_PATH, label_path=TRAIN_LABEL_PATH,
    #                         batch_size=4,
    #                         shuffle=True, n_classes=2, normalize_func=None)

    for index in range(gen.__len__()):
        _ = gen.__getitem__(index)
