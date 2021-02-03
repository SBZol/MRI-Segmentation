# coding:utf-8
"""
@time: 19-2-25
@author: zol
@contact: sbzol.chen@gmail.com
@file:  train_25d
@desc: None
@==============================@
@      _______ ____       __   @
@     / __/ _ )_  / ___  / /   @
@    _\ \/ _  |/ /_/ _ \/ /    @
@   /___/____//___/\___/_/     @
@                        SBZol @ 
@==============================@
"""
import sys
import os
import random
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from generator_25d import Generator4Train25D
from unet_25d import Unet25dCategorical
from loss_function import veins_loss, veins_dice0, veins_dice1, veins_dice2, veins_dice3, dice_coef_4
from dataIO import movefile
from pathvariable import *


class TrainVNet25D(object):
    def __init__(self, epochs=100, batch_size=1, image_size=(256, 256), img_c=1,
                 n_classes=2, normalize_func=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_h = image_size[0]
        self.img_w = image_size[1]
        self.img_c = img_c
        self.n_classes = n_classes
        self.normalize_func = normalize_func

    def plot_loss(self, res):
        """
        绘制loss

        Arguments:
            res -- modle 预测出的数据
        """
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), res.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), res.history["val_loss"], label="val_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig('./tensorboard_log/loss.png')
        plt.show()

        plt.plot(np.arange(0, self.epochs), res.history["acc"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), res.history["val_acc"], label="val_acc")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")

        plt.legend(loc="lower left")
        plt.savefig('./tensorboard_log/acc.png')
        plt.show()

    def get_generator(self):
        gen = Generator4Train25D(data_path=TRAIN_DATA_PATH,
                                 batch_size=self.batch_size,
                                 shuffle=True, n_classes=self.n_classes, normalize_func=self.normalize_func)
        return gen

    def get_val_generator(self):
        gen = Generator4Train25D(data_path=TRAIN_DATA_PATH,
                                 batch_size=self.batch_size,
                                 shuffle=True, n_classes=self.n_classes, normalize_func=self.normalize_func)
        return gen

    def train(self):
        """
        训练模型
        """
        print("---loading the Veins_UNet model...")
        # model = get_vnet_25d()
        model = Unet25dCategorical(img_h=32, img_w=32, img_c=1, n_label=4).get_unet()

        model.summary()

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',  # categorical_crossentropy
                      metrics=["accuracy", veins_loss, dice_coef_4, veins_dice0, veins_dice1, veins_dice2, veins_dice3])

        print("---UNet loading is done")

        plot_model(model, to_file='VNet_25D.png', show_shapes=True)

        with open('./model/VNet_25D.json', 'w') as files:
            files.write(model.to_json())

        # 保存的是最优模型结构和权重,
        model_checkpoint = ModelCheckpoint(r'./model/VNet_25D.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True, mode='min')

        # 定义early stop的条件
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # 生成log
        csv_logger = CSVLogger(r'./model/train_MRI_v_0225.csv', append=False)

        # 获取生成器对象
        generator_obj = self.get_generator()
        val_generator_obj = self.get_val_generator()

        print('Start to train the model...')
        res = model.fit_generator(
            generator=generator_obj,
            epochs=self.epochs,
            verbose=1,
            callbacks=[model_checkpoint, early_stopping, csv_logger],
            shuffle=False,
            max_queue_size=40,
            validation_data=val_generator_obj, )

        self.plot_loss(res)

    @classmethod
    def get_val_data(cls, proportion, train_root_path, val_root_path):
        image_list = os.listdir(train_root_path)
        random_list = random.sample(image_list, int(len(image_list) * proportion))

        for index, name in enumerate(random_list):
            movefile(srcfile=os.path.join(train_root_path, name),
                     dstfile=os.path.join(val_root_path, name))


if __name__ == '__main__':
    check_path()

    # TrainVNet25D.get_val_data(0.15, TRAIN_DATA_PATH, VAL_DATA_PATH)

    TrainVNet25D(epochs=50,
                 batch_size=4,
                 image_size=(32, 32),
                 img_c=1,
                 n_classes=4,
                 normalize_func=None).train()
