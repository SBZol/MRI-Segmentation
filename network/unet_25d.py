# coding:utf-8
"""
@time: 19-2-25
@author: zol
@contact: sbzol.chen@gmail.com
@file:  unet_25d
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
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model

from loss_function import dice_coef_4, dice_coef_loss

# os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['TENSORFLOW_FLAGS'] = 'mode=FAST_RUN, device=gpu, floatX=float32, \
#                                     optimizer=fast_compile,nvcc.flags=-D_FORCE_INLINES'


class Unet25dCategorical(object):

    def __init__(self, img_h=32, img_w=32, img_c=1, n_label=4):
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.n_label = n_label

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
        conv1 = Conv2D(filters=filter_num, kernel_size=(3, 3), padding='same', dilation_rate=(1, 1))(inputs)

        conv2 = Conv2D(filters=filter_num, kernel_size=(3, 3), padding='same', dilation_rate=(3, 3))(inputs)

        # conv3 = Conv2D(filters=filter_num, kernel_size=(3, 3), padding='same', dilation_rate=(5, 5))(inputs)

        merge = concatenate([conv1, conv2], axis=3)

        return merge

    def branch_down(self, inputs):
        # --- 1st stack and skip connection
        conv1 = self.dilation_conv_block(inputs, filter_num=4)
        conv1 = self.dilation_conv_block(conv1, filter_num=4)
        merge1 = concatenate([conv1, inputs], axis=3)  # cliff: skip connection
        pool1 = MaxPooling2D(pool_size=(2, 2))(merge1)

        # --- 2nd stack and skip connection
        conv2 = self.dilation_conv_block(pool1, filter_num=8)
        conv2 = self.dilation_conv_block(conv2, filter_num=8)
        merge2 = concatenate([conv2, pool1], axis=3)  # cliff: skip connection
        pool2 = MaxPooling2D(pool_size=(2, 2))(merge2)
        return pool2, merge1, merge2

    def get_unet(self):
        """
        获取网络模型,采用Dilation Convolution
        """

        sagittal_inputs = Input((self.img_h, self.img_h, self.img_c))
        coronal_inputs = Input((self.img_h, self.img_h, self.img_c))
        axial_inputs = Input((self.img_h, self.img_h, self.img_c))

        # Down-sampling pathway
        sagittal_ouput, merge1, merge2 = self.branch_down(sagittal_inputs)
        coronal_ouput, _, _ = self.branch_down(coronal_inputs)
        axial_ouput, _, _ = self.branch_down(axial_inputs)

        concat_all = concatenate([sagittal_ouput, coronal_ouput, axial_ouput], axis=3)

        # --- 3rd stack and skip connection
        conv3 = self.dilation_conv_block(concat_all, filter_num=150)
        conv3 = self.dilation_conv_block(conv3, filter_num=150)
        merge3 = concatenate([conv3, concat_all], axis=3)  # cliff: skip connection
        pool3 = MaxPooling2D(pool_size=(2, 2))(merge3)
        # pool3 = MaxPooling3D()(merge3)  # zol：适应128x128的输入

        # --- 4th stack and skip connection
        conv4 = self.dilation_conv_block(pool3, filter_num=300)
        conv4 = self.dilation_conv_block(conv4, filter_num=300)
        merge4 = concatenate([conv4, pool3], axis=3)  # cliff: skip connection
        pool4 = MaxPooling2D(pool_size=(2, 2))(merge4)

        # -------------------------------------
        # Bottom Stack
        # -------------------------------------
        # --- 5th stack without pooling and skip connection
        conv5 = self.dilation_conv_block(pool4, filter_num=600)
        conv5 = self.dilation_conv_block(conv5, filter_num=600)
        merge5 = concatenate([conv5, pool4], axis=3)  # cliff: skip connection
        # -------------------------------------
        # Bottom Stack
        # -------------------------------------

        # -------------------------------------
        # Up-sampling pathway
        # -------------------------------------

        # --- 6th stack and skip connection
        up1 = UpSampling2D(size=(2, 2))(merge5)

        up1 = Conv2D(128, (1, 1), padding='same', activation='selu', kernel_initializer='glorot_uniform')(up1)
        up1 = concatenate([up1, merge4], axis=3)
        conv6 = self.dilation_conv_block(up1, filter_num=300)
        conv6 = self.dilation_conv_block(conv6, filter_num=300)
        merge6 = concatenate([conv6, up1], axis=3)

        # --- 7th stack and skip connection
        up2 = UpSampling2D(size=(2, 2))(merge6)
        up2 = Conv2D(64, (1, 1), padding='same', activation='selu', kernel_initializer='glorot_uniform')(up2)
        up2 = concatenate([up2, merge3], axis=3)
        conv7 = self.dilation_conv_block(up2, filter_num=150)
        conv7 = self.dilation_conv_block(conv7, filter_num=150)
        merge7 = concatenate([conv7, up2], axis=3)

        # --- 8th stack and skip connection
        up3 = UpSampling2D(size=(2, 2))(merge7)
        up3 = Conv2D(32, (1, 1), padding='same', activation='selu', kernel_initializer='glorot_uniform')(up3)
        up3 = concatenate([up3, merge2], axis=3)
        conv8 = self.dilation_conv_block(up3, filter_num=75)
        conv8 = self.dilation_conv_block(conv8, filter_num=75)
        merge8 = concatenate([conv8, up3], axis=3)

        # --- 9th stack and skip connection
        up4 = UpSampling2D(size=(2, 2))(merge8)
        up4 = Conv2D(16, (1, 1), padding='same', activation='selu', kernel_initializer='glorot_uniform')(up4)
        up4 = concatenate([up4, merge1], axis=3)
        conv9 = self.dilation_conv_block(up4, filter_num=16)
        conv9 = self.dilation_conv_block(conv9, filter_num=16)
        merge9 = concatenate([conv9, up4], axis=3)

        # --- 10th output stack
        conv10 = Conv2D(2, (3, 3), padding='same', activation='selu', kernel_initializer='glorot_uniform')(merge9)
        conv11 = Conv2D(self.n_label, 1, activation='softmax')(conv10)

        # cliff 建立model
        model = Model(inputs=[sagittal_inputs, coronal_inputs, axial_inputs], outputs=conv11)
        # cliff 保存model示意图

        return model


if __name__ == '__main__':
    model = Unet25dCategorical().get_unet()
    plot_model(model, to_file='MRI_brain_seg_UNet25D.png', show_shapes=True)
