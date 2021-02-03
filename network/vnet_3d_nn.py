# _*_ coding:utf-8 _*_
"""
@time: 2018/6/20
@author: zol
@contact: 13012215283@sina.cn
@file: unet_3d_nn.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import os

from keras.layers import Conv3D, Input, concatenate, add, Softmax, Deconvolution3D, MaxPool3D, UpSampling3D
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.layers.core import Lambda
from keras import backend as K

from loss_function import dice_coef_4, dice_coef_loss

from functools import reduce
from operator import mul
import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TENSORFLOW_FLAGS'] = 'mode=FAST_RUN, device=gpu, floatX=float32, \
                                    optimizer=fast_compile,nvcc.flags=-D_FORCE_INLINES'


class VNet3D(object):
    def __init__(self, voxel_x=32, img_h=256, img_w=256, img_c=1, n_label=4):
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.n_label = n_label
        self.voxel_x = voxel_x

    @classmethod
    def get_num_params(cls):
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    @classmethod
    def downward_layer(cls, input_layer, n_convolutions, n_output_channels):
        inl = input_layer
        for _ in range(n_convolutions - 1):
            inl = PReLU()(
                Conv3D(n_output_channels // 2, (5, 5, 5), padding='same')(inl)
            )
        conv = Conv3D(n_output_channels // 2, (5, 5, 5), padding='same')(inl)
        add_d = add([conv, input_layer])

        downsample = Conv3D(filters=n_output_channels, kernel_size=(2, 2, 2), strides=(2, 2, 2))(add_d)
        prelu = PReLU()(downsample)
        return prelu, add_d

    @classmethod
    def contrastive_loss(cls, y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) +
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    @classmethod
    def upward_layer(cls, input0, input1, n_convolutions, n_output_channels, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        merged = concatenate([input0, input1], axis=4)
        inl = merged
        for _ in range(n_convolutions - 1):
            inl = PReLU()(
                Conv3D(n_output_channels * 4, (5, 5, 5), padding='same')(inl)
            )
        conv = Conv3D(n_output_channels * 4, (5, 5, 5), padding='same')(inl)
        add_u = add([conv, merged])

        upsample = Deconvolution3D(filters=n_output_channels, kernel_size=kernel_size, strides=strides)(add_u)
        return PReLU()(upsample)

    def vnet_struct(self, input_layer):
        # Layer 1
        conv_1 = Conv3D(16, (5, 5, 5), padding='same')(input_layer)
        repeat_1 = concatenate([input_layer] * 16)
        add_1 = add([conv_1, repeat_1])
        # prelu_1_1 = PReLU()(add_1)
        downsample_1 = Conv3D(filters=32, kernel_size=(1, 4, 4), strides=(1, 4, 4))(add_1)
        prelu_1_2 = PReLU()(downsample_1)

        # Layer 2,3,4
        out2, left2 = self.downward_layer(input_layer=prelu_1_2, n_convolutions=2, n_output_channels=64)
        out3, left3 = self.downward_layer(out2, 3, 128)
        out4, left4 = self.downward_layer(out3, 3, 256)

        # Layer 5
        conv_5_1 = Conv3D(256, (5, 5, 5), padding='same')(out4)
        prelu_5_1 = PReLU()(conv_5_1)
        conv_5_2 = Conv3D(256, (5, 5, 5), padding='same')(prelu_5_1)
        prelu_5_2 = PReLU()(conv_5_2)
        conv_5_3 = Conv3D(256, (5, 5, 5), padding='same')(prelu_5_2)
        add_5 = add([conv_5_3, out4])
        prelu_5_1 = PReLU()(add_5)

        downsample_5 = Deconvolution3D(filters=128, kernel_size=(2, 2, 2), strides=(2, 2, 2))(prelu_5_1)
        prelu_5_2 = PReLU()(downsample_5)

        # Layer 6,7,8
        out6 = self.upward_layer(input0=prelu_5_2, input1=left4, n_convolutions=3, n_output_channels=64)
        out7 = self.upward_layer(out6, left3, 3, 32)
        out8 = self.upward_layer(out7, left2, 2, 16, kernel_size=(1, 4, 4), strides=(1, 4, 4))

        # Layer 9
        merged_9 = concatenate([out8, add_1], axis=4)

        conv_9_1 = Conv3D(32, (5, 5, 5), padding='same')(merged_9)
        add_9 = add([conv_9_1, merged_9])

        conv_9_2 = Conv3D(4, (1, 1, 1), padding='same')(add_9)

        softmax = Softmax()(conv_9_2)

        return softmax

    def get_vnet(self):
        input_layer = Input(shape=(self.voxel_x, self.img_h, self.img_w, self.img_c), name='data')

        output = self.vnet_struct(input_layer)

        vet_model = Model(input_layer, output)

        vet_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
                          metrics=[dice_coef_4, dice_coef_loss, "accuracy"])

        return vet_model


class ACVNet(VNet3D):

    @classmethod
    def encoder_model(cls, input_shape):
        seq = Sequential()
        seq.add(Conv3D(filters=64, kernel_size=(3, 3, 3), input_shape=input_shape,
                       padding='same', activation='selu', kernel_initializer='lecun_normal'))
        seq.add(MaxPool3D(pool_size=(4, 4, 4)))
        seq.add(Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='selu',
                       kernel_initializer='lecun_normal'))
        seq.add(UpSampling3D(size=(4, 4, 4)))
        seq.add(Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='selu',
                       kernel_initializer='lecun_normal'))
        seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', activation='selu',
                       kernel_initializer='lecun_normal'))
        return seq

    @classmethod
    def euclidean_distance(cls, two_vects):
        # 两个向量的欧氏距离计算
        x, y = two_vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    @classmethod
    def eucl_dist_output_shape(cls, shapes):
        # 欧氏距离矩阵的 shape
        shape1, shape2 = shapes
        return shape1[0], 1

    def get_ac_vnet(self):
        input_layer = Input(shape=(self.voxel_x, self.img_h, self.img_w, self.img_c), name='data')
        vnet_output = self.vnet_struct(input_layer)

        train_label = Input(shape=(self.voxel_x, self.img_h, self.img_w, self.n_label), name="train_label")
        base_seq_model = self.encoder_model(input_shape=(self.voxel_x, self.img_h, self.img_w, self.n_label))

        encoder1 = base_seq_model(train_label)
        encoder2 = base_seq_model(vnet_output)
        distance = Lambda(function=self.euclidean_distance, output_shape=self.eucl_dist_output_shape)(
            [encoder1, encoder2])
        model = Model(inputs=[input_layer, train_label], outputs=[vnet_output, distance])
        model.compile(optimizer=Adam(lr=0.0001), loss=['categorical_crossentropy', self.contrastive_loss])

        print('model compile')
        return model


if __name__ == '__main__':
    # model = VNet3D(voxel_x=32, img_h=256, img_w=256, img_c=1,
    #                n_label=4).get_vnet()
    # plot_model(model, to_file='MRI_brain_seg_VNet3D.png', show_shapes=True)

    model = ACVNet(voxel_x=32, img_h=256, img_w=256, img_c=1,
                   n_label=4).get_ac_vnet()
    plot_model(model, to_file='MRI_brain_seg_AC_VNet3D.png', show_shapes=True)
