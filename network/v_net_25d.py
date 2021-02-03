# coding:utf-8
"""
@time: 19-1-23
@author: zol
@contact: sbzol.chen@gmail.com
@file:  v_net_2.5d
@desc: None
@==============================@
@      _______ ____       __   @
@     / __/ _ )_  / ___  / /   @
@    _\ \/ _  |/ /_/ _ \/ /    @
@   /___/____//___/\___/_/     @
@                        SBZol @ 
@==============================@
"""
import numpy as np
from keras.layers import Input, concatenate, PReLU, add, Deconvolution2D, Softmax
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.utils.vis_utils import plot_model


def downward_layer(input_layer, convs_count, output_channels, name, down_kernal_size=(2, 2), down_strides=(2, 2)):
    inl = input_layer
    for _ in range(convs_count):
        inl = PReLU()(
            Conv2D(output_channels // 2, (3, 3), padding='same')(inl)
        )

    add_d = add([inl, input_layer])

    down_sample = Conv2D(filters=output_channels,
                         kernel_size=down_kernal_size,
                         strides=down_strides,
                         name=name)(add_d)

    p_relu = PReLU()(down_sample)
    return p_relu, add_d


def upward_layer(input0, input1, convs_count, output_channels, name, kernel_size=(2, 2), strides=(2, 2)):
    merged = concatenate([input0, input1], axis=3)
    inl = merged
    for _ in range(convs_count):
        inl = PReLU()(
            Conv2D(output_channels * 4, (3, 3), padding='same')(inl)
        )
    add_u = add([inl, merged])

    upsample = Deconvolution2D(filters=output_channels,
                               kernel_size=kernel_size,
                               strides=strides,
                               name=name)(add_u)
    return PReLU()(upsample)


def branch_down(axial_input, name=''):
    # axial layer 1
    conv_1 = Conv2D(16, (3, 3), padding='same')(axial_input)
    # repeat_1 = concatenate([axial_input] * 16)
    repeat_1 = Conv2D(16, (1, 1))(axial_input)
    axial_left1 = add([conv_1, repeat_1])

    down_sample = Conv2D(filters=32,
                         kernel_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         name=name + '_down_layer_1')(axial_left1)

    out1 = PReLU()(down_sample)

    inl = out1
    for _ in range(2):
        inl = PReLU()(
            Conv2D(32, (3, 3), padding='same')(inl)
        )

    add_d = add([inl, out1])

    return add_d


def get_vnet_25d():
    # sagittal branch
    sagittal_input = Input(shape=(32, 32, 1), name='sagittal_input')
    sagittal_out = branch_down(sagittal_input, name='sagittal')

    # coronal branch
    coronal_input = Input(shape=(32, 32, 1), name='coronal_input')
    coronal_out = branch_down(coronal_input, name='coronal')

    # axial branch
    axial_input = Input(shape=(32, 32, 1), name='axial_input')
    axial_out = branch_down(axial_input, name='axial')

    share_concate = concatenate([sagittal_out, coronal_out, axial_out])

    # share layer 1, 2 (down)
    share_out1, share_left1 = downward_layer(share_concate, 3, 192, name='share_down_layer_1')
    share_out2, share_left2 = downward_layer(share_out1, 3, 384, name='share_down_layer_2')

    # share layer 3 (bottom)
    conv_bottom_1 = Conv2D(384, (3, 3), padding='same')(share_out2)
    prelu_bottom_1 = PReLU()(conv_bottom_1)
    conv_bottom__2 = Conv2D(384, (3, 3), padding='same')(prelu_bottom_1)
    prelu_bottom_2 = PReLU()(conv_bottom__2)
    conv_bottom__3 = Conv2D(384, (3, 3), padding='same')(prelu_bottom_2)
    prelu_bottom_3 = PReLU()(conv_bottom__3)
    add_bottom = add([prelu_bottom_3, share_out2])

    downsample_bottom = Deconvolution2D(filters=192,
                                        kernel_size=(2, 2),
                                        strides=(2, 2),
                                        name='share_up_layer_1')(add_bottom)
    share_out3 = PReLU()(downsample_bottom)

    # share layer 4,5 (up)
    share_out4 = upward_layer(share_out3, share_left2, 3, 96, name='share_up_layer_2')

    share_out5 = upward_layer(share_out4, share_left1, 3, 48, name='share_up_layer_3')

    axial_conv_last = Conv2D(4, (1, 1), padding='same')(share_out5)
    axial_ouput = Softmax()(axial_conv_last)

    model = Model(inputs=[axial_input, coronal_input, sagittal_input], outputs=axial_ouput)
    plot_model(model, to_file='v_net_2.5d.png', show_shapes=True)

    return model


if __name__ == '__main__':
    get_vnet_25d()
