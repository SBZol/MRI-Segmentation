# _*_ coding:utf-8 _*_
"""
@time: 2018/6/22
@author: zol
@contact: 13012215283@sina.cn
@file: train.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import random

from unet_3d_nn import Unet3dCategorical
from vnet_3d_nn import VNet3D
from pathvariable import *
from preprocess import PreProcess4SSpacing, PreProcess4MSpacing
from generator import Generator4Train3D
from create_train_data import CreateTrainData, CreateTrainData4SS
from dataIO import movefile


class Train(object):
    def __init__(self, epochs=100, batch_size=1, gen_size=(32, 256, 256), img_c=1,
                 n_label=4):
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_h = gen_size[1]
        self.img_w = gen_size[2]
        self.img_c = img_c
        self.n_label = n_label
        self.voxel_x = gen_size[0]
        self.gen_size = gen_size

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
        gen = Generator4Train3D(data_path=TRAIN_DATA_PATH, batch_size=self.batch_size,
                                shuffle=True,
                                n_classes=self.n_label,
                                normalize_func=PreProcess4SSpacing.standardiztion)
        return gen

    @classmethod
    def get_val_data(cls, proportion, train_root_path, val_root_path):
        image_list = os.listdir(train_root_path)
        random_list = random.sample(image_list, int(len(image_list) * proportion))

        for index, name in enumerate(random_list):
            movefile(srcfile=os.path.join(train_root_path, name),
                     dstfile=os.path.join(val_root_path, name))


class TrainUNet3D(Train):

    def train(self):
        """
        训练模型
        """
        print("---loading the UNet3D model...")
        model = Unet3dCategorical(voxel_x=self.voxel_x, img_h=self.img_h, img_w=self.img_w, img_c=self.img_c,
                                  n_label=self.n_label).get_unet()
        print("---UNet3D loading is done")

        plot_model(model, to_file='MRI_brain_seg_UNet3D.png', show_shapes=True)

        # cliff 保存model结构为json文件
        with open('./model/MRI_brain_3DUet_seg.json', 'w') as files:
            files.write(model.to_json())

        # 保存的是最优模型结构和权重,
        model_checkpoint = ModelCheckpoint(r'./model/MRI_brain_seg_UNet3D.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True, mode='min')

        # 将loss ，acc， val_loss ,val_acc 记录 tensorboard
        # cliff 在CMD窗口执行 tensorboard --logdir = log_path运行tensorboard服务器，根据CMD提示的本地网址在浏览器产看结果
        logdir = "./tensorboard_log"
        tensorboard = TensorBoard(log_dir=logdir)

        # 定义early stop的条件
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        # 获取生成器对象
        generator_obj = self.get_generator()
        val_generator_obj = self.get_generator()

        print('Start to train the model...')
        res = model.fit_generator(
            generator=generator_obj,
            epochs=self.epochs,
            verbose=1,
            callbacks=[model_checkpoint, early_stopping],
            shuffle=False,
            max_queue_size=40,
            validation_data=val_generator_obj, )

        self.plot_loss(res)


class TrainVNet3D(Train):

    def train(self):
        """
        训练模型
        """
        print("---loading the VNet3D model...")
        model = VNet3D(voxel_x=self.voxel_x, img_h=self.img_h, img_w=self.img_w, img_c=self.img_c,
                       n_label=self.n_label).get_vnet()
        print("---VNet3D loading is done")

        plot_model(model, to_file='MRI_brain_seg_VNet3D.png', show_shapes=True)

        with open('./model/MRI_brain_3DVNet_seg.json', 'w') as files:
            files.write(model.to_json())

        # 保存的是最优模型结构和权重,
        model_checkpoint = ModelCheckpoint(r'./model/MRI_brain_seg_VNet3D.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True, mode='min')

        # 将loss ，acc， val_loss ,val_acc 记录 tensorboard
        # 在CMD窗口执行 tensorboard --logdir = log_path运行tensorboard服务器，根据CMD提示的本地网址在浏览器产看结果
        logdir = "./tensorboard_log"
        tensorboard = TensorBoard(log_dir=logdir)

        # 定义early stop的条件
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # 获取生成器对象
        generator_obj = self.get_generator()
        val_generator_obj = self.get_generator()

        print('Start to train the model...')
        res = model.fit_generator(
            generator=generator_obj,
            epochs=self.epochs,
            verbose=1,
            callbacks=[model_checkpoint, early_stopping],
            shuffle=False,
            max_queue_size=40,
            validation_data=val_generator_obj, )

        self.plot_loss(res)


if __name__ == '__main__':
    """
    在这里训练模型, step1 和 step2 运行后生成训练需要的数据
    前2个步骤执行完后就已经为训练做好了准备,可以把step1 - setp2 注释掉, 只执行 setp3 来训练.
    整个过程之后会慢慢优化
    """
    network = 'u-net'  # v-net or u-net
    img_name = 'mprage_anonymized.nii'
    label_name = 'Brain_4_labels.nii'
    # 检测文件夹路径是否存在，不存在则创建
    check_path()

    # # step1：生成 train set
    # train_data_paths = CreateTrainData.get_original_data_paths(ORIGINAL_PATH, 'Brain_4_labels.nii')
    #
    # CreateTrainData4SS(train_data_paths=train_data_paths,
    #                    output_path=TRAIN_DATA_PATH,
    #                    nii_ouput_path=TRAIN_DATA_NII_PATH,
    #                    target_shape=(32, 256, 256),
    #                    step=16,
    #                    num_classes=4).create_train_data()
    # # setp 2: 分出验证集
    # Train.get_val_data(0.10, TRAIN_DATA_PATH, VAL_DATA_PATH)

    # step3：训练模型
    if network == 'u-net':
        TrainUNet3D(epochs=400, gen_size=(32, 256, 256), batch_size=1, img_c=1,
                    n_label=4, ).train()

    if network == 'v-net':
        TrainVNet3D(epochs=150,
                    gen_size=(32, 256, 256),
                    img_c=1,
                    n_label=4,
                    ).train()
