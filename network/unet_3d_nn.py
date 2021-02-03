# _*_ coding:utf-8 _*_
"""
@time: 2018/6/20
@author: zol
@contact: 13012215283@sina.cn
@file: unet_3d_nn.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import os
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model

from loss_function import dice_coef_4, dice_coef_loss


os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TENSORFLOW_FLAGS'] = 'mode=FAST_RUN, device=gpu, floatX=float32, \
                                    optimizer=fast_compile,nvcc.flags=-D_FORCE_INLINES'


class Unet3dCategorical(object):

    def __init__(self, voxel_x=32, img_h=256, img_w=256, img_c=1, n_label=4):
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.n_label = n_label
        self.voxel_x = voxel_x

    @classmethod
    def dilation_conv_block(cls, inputs, filter_num):
        """
        使用扩张(空洞)3D卷积，连续三层不同dilation_rate的空洞卷积，增加感受野的丰富度
        dilation_rate的优化选择可参考《如何理解空洞卷积（dilated convolution）？ - 知乎.pdf》Page3

        Arguments:
            inputs -- 卷积的输入
            filter_num -- 卷积filter的数量

        Returns:
            merge -- 多层卷积的 concatenate
        """

        conv1 = Conv3D(filters=filter_num, kernel_size=(3, 3, 3), padding='same', dilation_rate=(1, 1, 1),
                       activation='selu', kernel_initializer='glorot_uniform')(inputs)
        conv2 = Conv3D(filters=filter_num, kernel_size=(3, 3, 3), padding='same', dilation_rate=(1, 3, 3),
                       activation='selu', kernel_initializer='glorot_uniform')(inputs)
        conv3 = Conv3D(filters=filter_num, kernel_size=(3, 3, 3), padding='same', dilation_rate=(1, 5, 5),
                       activation='selu', kernel_initializer='glorot_uniform')(inputs)
        conv4 = Conv3D(filters=filter_num, kernel_size=(3, 3, 3), padding='same', dilation_rate=(1, 7, 7),
                       activation='selu', kernel_initializer='glorot_uniform')(inputs)
        merge = concatenate([conv1, conv2, conv3, conv4], axis=4)

        return merge

    def get_unet(self):
        """
        获取网络模型,采用Dilation Convolution
        """

        inputs = Input((self.voxel_x, self.img_h, self.img_h, self.img_c))

        # -------------------------------------
        # Down-sampling pathway
        # -------------------------------------
        # --- 1st stack and skip connection
        conv1 = self.dilation_conv_block(inputs, filter_num=4)
        conv1 = self.dilation_conv_block(conv1, filter_num=4)
        merge1 = concatenate([conv1, inputs], axis=4)  # cliff: skip connection
        pool1 = MaxPooling3D(pool_size=(4, 4, 4))(merge1)

        # --- 2nd stack and skip connection
        conv2 = self.dilation_conv_block(pool1, filter_num=8)
        conv2 = self.dilation_conv_block(conv2, filter_num=8)
        merge2 = concatenate([conv2, pool1], axis=4)  # cliff: skip connection
        pool2 = MaxPooling3D(pool_size=(2, 4, 4))(merge2)
        # pool2 = MaxPooling3D()(merge2)  # zol：适应128x128的输入

        # --- 3rd stack and skip connection
        conv3 = self.dilation_conv_block(pool2, filter_num=16)
        conv3 = self.dilation_conv_block(conv3, filter_num=16)
        merge3 = concatenate([conv3, pool2], axis=4)  # cliff: skip connection
        pool3 = MaxPooling3D(pool_size=(2, 4, 4))(merge3)
        # pool3 = MaxPooling3D()(merge3)  # zol：适应128x128的输入

        # --- 4th stack and skip connection
        conv4 = self.dilation_conv_block(pool3, filter_num=32)
        conv4 = self.dilation_conv_block(conv4, filter_num=32)
        merge4 = concatenate([conv4, pool3], axis=4)  # cliff: skip connection
        pool4 = MaxPooling3D(pool_size=(2, 4, 4))(merge4)
        # pool4 = MaxPooling3D()(merge4)  # zol：适应128x128的输入

        # -------------------------------------
        # Bottom Stack
        # -------------------------------------
        # --- 5th stack without pooling and skip connection
        conv5 = self.dilation_conv_block(pool4, filter_num=64)
        conv5 = self.dilation_conv_block(conv5, filter_num=64)
        merge5 = concatenate([conv5, pool4], axis=4)  # cliff: skip connection
        # -------------------------------------
        # Bottom Stack
        # -------------------------------------

        # -------------------------------------
        # Up-sampling pathway
        # -------------------------------------

        # --- 6th stack and skip connection
        up1 = UpSampling3D(size=(2, 4, 4))(merge5)
        # up1 = UpSampling3D(size=(2, 2, 2))(merge5)  # zol：适应128x128的输入
        up1 = Conv3D(128, (1, 1, 1), padding='same', activation='selu', kernel_initializer='glorot_uniform')(up1)
        up1 = concatenate([up1, merge4], axis=4)
        conv6 = self.dilation_conv_block(up1, filter_num=32)
        conv6 = self.dilation_conv_block(conv6, filter_num=32)
        merge6 = concatenate([conv6, up1], axis=4)

        # --- 7th stack and skip connection
        up2 = UpSampling3D(size=(2, 4, 4))(merge6)
        # up2 = UpSampling3D(size=(2, 2, 2))(merge6)  # zol：适应128x128的输入
        up2 = Conv3D(64, (1, 1, 1), padding='same', activation='selu', kernel_initializer='glorot_uniform')(up2)
        up2 = concatenate([up2, merge3], axis=4)
        conv7 = self.dilation_conv_block(up2, filter_num=16)
        conv7 = self.dilation_conv_block(conv7, filter_num=16)
        merge7 = concatenate([conv7, up2], axis=4)

        # --- 8th stack and skip connection
        up3 = UpSampling3D(size=(2, 4, 4))(merge7)
        # up3 = UpSampling3D(size=(2, 2, 2))(merge7)  # zol：适应128x128的输入
        up3 = Conv3D(32, (1, 1, 1), padding='same', activation='selu', kernel_initializer='glorot_uniform')(up3)
        up3 = concatenate([up3, merge2], axis=4)
        conv8 = self.dilation_conv_block(up3, filter_num=8)
        conv8 = self.dilation_conv_block(conv8, filter_num=8)
        merge8 = concatenate([conv8, up3], axis=4)

        # --- 9th stack and skip connection
        up4 = UpSampling3D(size=(4, 4, 4))(merge8)
        up4 = Conv3D(16, (1, 1, 1), padding='same', activation='selu', kernel_initializer='glorot_uniform')(up4)
        up4 = concatenate([up4, merge1], axis=4)
        conv9 = self.dilation_conv_block(up4, filter_num=4)
        conv9 = self.dilation_conv_block(conv9, filter_num=4)
        merge9 = concatenate([conv9, up4], axis=4)

        # --- 10th output stack
        conv10 = Conv3D(2, (3, 3, 3), padding='same', activation='selu', kernel_initializer='glorot_uniform')(merge9)
        conv11 = Conv3D(self.n_label, 1, activation='softmax')(conv10)

        # crf_layer = CRF(units=4, sparse_target=True)
        # crf = crf_layer(conv11)

        # cliff 建立model
        model = Model(inputs=inputs, outputs=conv11)
        # cliff 保存model示意图

        # cliff 编译model
        # parallel_model = multi_gpu_model(model, gpus=4)。
        # model.compile(optimizer=Adam(lr=0.0001), loss=self.dice_loss, metrics=["accuracy"])
        model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
                      metrics=[dice_coef_4, dice_coef_loss,"accuracy"])

        return model


if __name__ == '__main__':
    model = Unet3dCategorical().get_unet()
    plot_model(model, to_file='MRI_brain_seg_UNet3D.png', show_shapes=True)
