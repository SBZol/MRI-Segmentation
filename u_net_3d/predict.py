# _*_ coding:utf-8 _*_
"""
@time: 2018/6/22
@author: zol
@contact: 13012215283@sina.cn
@file: predict.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import sys
import os
import getopt
import math
import copy
import numpy as np
from keras import models
from dltk.core.metrics import dice

from generator import GeneratorSSpacing, GeneratorMSpacing
from preprocess import PreProcess4SSpacing, PreProcess4MSpacing
from utils import np_2_nii, dicom2nii, get_main_axis, TimeCalculation, compress_nii
from nilearn import image
from dipy.align.reslice import reslice

from postprocess import Postprocess

# 屏蔽 tensorflow 的通知信息和警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class _Predict(object):
    def __init__(self, mod_path, dicom_path, nii_path, label_path, output_path, model_names):
        """
        构造方法

        Arguments:
            mod_path -- modle 的路径
            dicom_path -- 预测的 dicom 文件夹路径（如果 nii_path, 不为 None 则该参数无效）
            nii_path -- 预测的 nii 文件路径（可选,如果该参数有值,则会忽略 dicom_path, 直接用 nii_path 读取数据）
            label_path -- 用于计算 dice 值的 label, 文件格式为 nii（可选）
            output_path -- 预测输出结果存放路径
            input_x_size -- 网络输入 x 轴的大小
            model_names -- 一个存储载入模型时需要的json和h5文件名的名字 例如['MRI_brain_3DUet_seg.json', 'MRI_brain_seg_UNet3D.h5']
        """
        self.mod_path = mod_path
        self.dicom_path = dicom_path
        self.nii_path = nii_path
        self.label_path = label_path
        self.output_path = output_path
        self.standard_shape = (32, 256, 256)
        self.model_names = model_names

    def predict(self):
        pass

    def load_model(self):
        """
        根据模型路径,json 文件名和 h5 文件名载入模型
        :return: model
        """
        with open(os.path.join(self.mod_path, self.model_names[0])) as file:
            model = models.model_from_json(file.read())
        model.load_weights(os.path.join(self.mod_path, self.model_names[1]))
        return model

    def output_nii(self, output_nii, origin_nii, label_num=-1, compress=True):
        """
        输出 nii 图像到保存路径下

        Arguments:
            label_num -- 选定标签 2:灰质, 3:白质, 4：脑脊液, -1:全部
        """
        names = {'-1': 'All.nii', '1': 'GrayMatter.nii', '2': 'WhiteMatter.nii', '3': 'CSF.nii'}
        print('------> nii dim:' + str(output_nii.header['dim']))
        print('------> nii pixdim:' + str(output_nii.header['pixdim']))

        # 这里必须用copy,不然改变了原始的输入会对接下来的输出造成影响
        output_nii_data = copy.deepcopy(output_nii).get_data()
        origin_main_axis = get_main_axis(origin_nii.get_data())
        transpose_shape_dict = {0: (1, 2, 0), 1: (0, 2, 1), 2: (0, 1, 2)}

        res_nii_data = np.transpose(output_nii_data, transpose_shape_dict[origin_main_axis])

        # 紧急处理,转换左右手坐标系
        if origin_main_axis == 2:
            res_nii_data = res_nii_data[::-1, :, :]

        print('------> output shape: ' + str(res_nii_data.shape) + ' transpose touple ' + str(
            transpose_shape_dict[origin_main_axis]))

        # 调换了坐标轴,因此需要改变dim 和pixdim
        oringin_pixdim = output_nii.header['pixdim']
        new_header = copy.deepcopy(output_nii.header)
        new_header['dim'][1] = res_nii_data.shape[0]
        new_header['dim'][2] = res_nii_data.shape[1]
        new_header['dim'][3] = res_nii_data.shape[2]

        pixdim_shape_dict = {0: (2, 3, 1), 1: (1, 3, 2), 2: (1, 2, 3)}

        new_header['pixdim'][1] = oringin_pixdim[pixdim_shape_dict[origin_main_axis][0]]
        new_header['pixdim'][2] = oringin_pixdim[pixdim_shape_dict[origin_main_axis][1]]
        new_header['pixdim'][3] = oringin_pixdim[pixdim_shape_dict[origin_main_axis][2]]

        if label_num == -1:
            path = os.path.join(self.output_path, names[str(label_num)])
            np_2_nii(origin_nii, res_nii_data, path, True, header=new_header)
            # np_2_nii(origin_nii, output_nii_data, path, True, header=None)
            if compress:
                compress_nii(path, path + '.gz', remove_origin=True)
            print('------> output all label')
        else:
            output_nii_data[output_nii_data != label_num] = 0
            path = os.path.join(self.output_path, names[str(label_num)])
            np_2_nii(origin_nii, res_nii_data, path, True, header=new_header)
            # np_2_nii(origin_nii, output_nii_data, path, True, header=None)
            if compress:
                compress_nii(path, path + '.gz', remove_origin=True)
            print('------> output label ' + str(label_num))

    def calculate_dice(self, predict_nii):
        """
        根据传预测输出的 nii 和 Label 计算 Dice
        :param predict_nii: 预测出来用于计算 Dice 的 nii
        :return: 保存各类 Dice 值的数组 -> [背景Dice, 灰质Dice, 白质Dice, 脑脊液Dice]
        """
        if self.label_path:
            print('------------>> step7: calculate dice...')
            label_data = image.load_img(self.label_path).get_data()

            # 一些奇怪的label数据得到的不是整数,如输出值是2,确变成了1.999999,这里需要四舍五入
            label_data = np.rint(label_data)

            predict_data = image.load_img(predict_nii).get_data()
            num_classes = len(np.unique(label_data))
            print(np.unique(label_data))
            print(num_classes)
            dict_arr = dice(predictions=predict_data, labels=label_data, num_classes=num_classes)
            print(dict_arr)
            return dict_arr


class StaticSpacingPredict(_Predict):

    def predict(self):
        time_preprocess = TimeCalculation(name='Preprocess')
        time_predict = TimeCalculation(name='Predict')
        time_postprocess = TimeCalculation(name='Postprocess')
        time_total = TimeCalculation(name='Total')

        # 总计时开始
        time_total.begin()

        # 预处理计时开始
        time_preprocess.begin()

        # dicom 转换成 nii, 如果 nii_path 有值, 则默认不需要转换，直接去 nii_path 作为 origin_nii_path
        print('------------>> step1: dicom2nii...')

        origin_nii_path = self.nii_path if self.nii_path else dicom2nii(dicom_path=self.dicom_path,
                                                                        save_path=self.output_path)

        origin_nii = image.load_img(origin_nii_path)

        print('------> origin nii shape: ' + str(origin_nii.get_data().shape))

        # 预处理
        print('------------>> step2: preprocrss...')
        r_nii_data, reslice_data, reslice_affine, main_axis, padding = PreProcess4SSpacing.preprocess(origin_nii,
                                                                                                      self.standard_shape,
                                                                                                      normalize='standardiztion')
        x_padding = padding[0]
        y_padding = padding[1]
        z_padding = padding[2]

        # 预处理计时结束
        time_preprocess.end()

        # 预测计时开始
        time_predict.begin()

        # 载入模型
        print('------------>> step3: load model...')
        model = self.load_model()

        # 开始预测
        print('------------>> step4: start to predict...')

        # 计算分块取数据需要的步骤数
        step = int(math.ceil(r_nii_data.shape[0] / self.standard_shape[0]))
        print('------> generator steps: ' + str(step))

        pre_imgs = model.predict_generator(
            generator=GeneratorSSpacing(normalize_func=PreProcess4MSpacing.standardiztion).pre_generator(r_nii_data),
            steps=step)
        # 预测计时结束
        time_predict.end()

        # 后处理计时开始
        time_postprocess.begin()

        output_imgs = np.argmax(pre_imgs, axis=4)

        # 巨坑：输出图像前把类型转换成float32,不然MRIcorN显示的时候会出现问题
        output_imgs = output_imgs.astype(np.float32)
        print('------> output image shape: ' + str(output_imgs.shape))
        print('------> unique of output image: ' + str(np.unique(output_imgs)))

        # 输出图像并保存
        print('------------>> step5: stitching_predicted_data...')
        stitching_data = self.stitching_predicted_data(output_imgs,
                                                       remainder=(r_nii_data.shape[0] % self.standard_shape[0]))

        # 还原 padding 的数据
        print(reslice_data.shape)
        stitching_data = stitching_data[:(stitching_data.shape[0] - x_padding[1]),
                         :(stitching_data.shape[1] - y_padding[1]),
                         :(stitching_data.shape[2] - z_padding[1])]

        print('------> not padding output shape: ' + str(stitching_data.shape))

        # 主轴还原
        transpose_shape_dict = {0: (0, 1, 2), 1: (1, 0, 2), 2: (1, 2, 0)}
        stitching_data = np.transpose(stitching_data, transpose_shape_dict[main_axis])

        reslice_up_data, reslice_up_affine = reslice(stitching_data, reslice_affine, (1., 1., 1.),
                                                     origin_nii.header.get_zooms()[:3], mode='nearest', order=0)
        print(np.unique(reslice_up_data))
        print('------> reslice_up shape: ' + str(reslice_up_data.shape))

        predict_nii = np_2_nii(origin_nii_path, reslice_up_data, need_save=False)

        print('------------>> step6: postprocess...')
        Postprocess(predict_nii, origin_nii, origin_nii_path, self.output_path).postprocess()

        # 输出 nii 图像
        print('------> output nii file...')
        self.output_nii(predict_nii, origin_nii)
        self.output_nii(predict_nii, origin_nii, 1)
        self.output_nii(predict_nii, origin_nii, 2)
        self.output_nii(predict_nii, origin_nii, 3)

        # 计算 dice 值
        self.calculate_dice(predict_nii)

        # 后处理计时结束
        time_postprocess.end()

        # 总计时结束
        time_total.end()
        print('------------>> Done!')

    @classmethod
    def stitching_predicted_data(cls, imgs, remainder):
        """
        把预测到的数据转换成 nii 保存

        Arguments:
            imgs -- 预测的图像数据
            oringin_path -- 原始的图像路径
            remainder -- 原始图像按照分块预测, 分块后最后的余数. 例如 input_x_size=32, 输入图像主轴大小为65, 则remainder=1
        """
        print('------> remainder: ' + str(remainder))

        concatenate_vol = imgs[0, :, :, :]
        range_val = imgs.shape[0] - 1
        print(concatenate_vol.shape)
        for i in range(range_val):
            if (i < (range_val - 1)) or (i == (range_val - 1) and remainder == 0):
                concatenate_vol = np.concatenate((concatenate_vol, imgs[i + 1, :, :, :]))
                print(concatenate_vol.shape)
            else:
                shape = imgs.shape[1]
                concatenate_vol = np.concatenate(
                    (concatenate_vol, imgs[i + 1, (shape - remainder):shape, :, :]))
                print(concatenate_vol.shape)

        output = concatenate_vol
        print('------> concatenate_vol.shape: ' + str(output.shape))

        return output


class MutabelSpacingPredict(_Predict):
    def predict(self):
        time_preprocess = TimeCalculation(name='Preprocess')
        time_predict = TimeCalculation(name='Predict')
        time_postprocess = TimeCalculation(name='Postprocess')
        time_total = TimeCalculation(name='Total')

        # 总计时开始
        time_total.begin()

        # 预处理计时开始
        time_preprocess.begin()

        # dicom 转换成 nii, 如果 nii_path 有值, 则默认不需要转换，直接去 nii_path 作为 origin_nii_path
        print('------------>> step1: dicom2nii...')

        origin_nii_path = self.nii_path if self.nii_path else dicom2nii(dicom_path=self.dicom_path,
                                                                        save_path=self.output_path)

        origin_nii = image.load_img(origin_nii_path)

        print('------> origin nii shape: ' + str(origin_nii.get_data().shape))

        # 预处理
        print('------------>> step2: preprocrss...')
        r_nii_data, main_axis, padding = PreProcess4MSpacing.preprocess(origin_nii,
                                                                        self.standard_shape,
                                                                        normalize='standardiztion')
        x_padding = padding[0]
        y_padding = padding[1]
        z_padding = padding[2]

        # 预处理计时结束
        time_preprocess.end()

        # 预测计时开始
        time_predict.begin()

        # 载入模型
        print('------------>> step3: load model...')
        model = self.load_model()

        # 开始预测
        print('------------>> step4: start to predict...')

        # 计算分块取数据需要的步骤数
        step = int(math.ceil(r_nii_data.shape[0] / self.standard_shape[0]))
        print('------> generator steps: ' + str(step))

        pre_imgs = model.predict_generator(
            generator=GeneratorMSpacing(normalize_func=PreProcess4MSpacing.standardiztion).pre_generator(r_nii_data),
            steps=step)
        # 预测计时结束
        time_predict.end()

        # 后处理计时开始
        time_postprocess.begin()

        output_imgs = np.argmax(pre_imgs, axis=4)

        # 巨坑：输出图像前把类型转换成float32,不然MRIcorN显示的时候会出现问题
        output_imgs = output_imgs.astype(np.float32)
        print('------> output image shape: ' + str(output_imgs.shape))
        print('------> unique of output image: ' + str(np.unique(output_imgs)))

        # 输出图像并保存
        print('------------>> step5: stitching_predicted_data...')
        stitching_data = self.stitching_predicted_data(output_imgs, self.standard_shape)

        # 还原 padding 的数据
        print(stitching_data.shape)
        stitching_data = stitching_data[:(stitching_data.shape[0] - x_padding[1]),
                         :(stitching_data.shape[1] - y_padding[1]),
                         :(stitching_data.shape[2] - z_padding[1])]

        print('------> not padding output shape: ' + str(stitching_data.shape))

        # 主轴还原
        transpose_shape_dict = {0: (0, 1, 2), 1: (1, 0, 2), 2: (1, 2, 0)}
        stitching_data = np.transpose(stitching_data, transpose_shape_dict[main_axis])

        print(np.unique(stitching_data))

        predict_nii = np_2_nii(origin_nii_path, stitching_data, need_save=False)

        print('------------>> step6: postprocess...')
        Postprocess(predict_nii, origin_nii, origin_nii_path, self.output_path).postprocess()

        # 输出 nii 图像
        print('------> output nii file...')
        self.output_nii(predict_nii, origin_nii)
        self.output_nii(predict_nii, origin_nii, 1)
        self.output_nii(predict_nii, origin_nii, 2)
        self.output_nii(predict_nii, origin_nii, 3)

        # 计算 dice 值
        self.calculate_dice(predict_nii)

        # 后处理计时结束
        time_postprocess.end()

        # 总计时结束
        time_total.end()
        print('------------>> Done!')

    @classmethod
    def stitching_predicted_data(cls, imgs, target_shape):
        """
        把预测到的数据转换成 nii 保存
        :param imgs:
        :param target_shape:
        :return:
        """

        def get_axisnum_and_range(axis):
            axis_num = target_shape[axis]
            range = int(math.ceil(axis_num / imgs.shape[axis + 1]))
            return axis_num, range

        def get_start_end_num(step, axis):
            start_num = step * imgs.shape[axis + 1]
            end_num = (step + 1) * imgs.shape[axis + 1]
            return start_num, end_num

        axis_num_x, range_x = get_axisnum_and_range(0)
        axis_num_y, range_y = get_axisnum_and_range(1)
        axis_num_z, range_z = get_axisnum_and_range(2)
        size = (imgs.shape[1], imgs.shape[2], imgs.shape[3])
        num = 0
        output = np.zeros(target_shape)
        for i_y in range(range_y):
            for i_z in range(range_z):
                for i_x in range(range_x):
                    x_start_num, x_end_num = get_start_end_num(i_x, 0)
                    y_start_num, y_end_num = get_start_end_num(i_y, 1)
                    z_start_num, z_end_num = get_start_end_num(i_z, 2)

                    output[
                    x_start_num if x_end_num <= axis_num_x else (
                            axis_num_x - size[0]): x_end_num if x_end_num <= axis_num_x else axis_num_x,
                    y_start_num if y_end_num <= axis_num_y else (
                            axis_num_y - size[1]): y_end_num if y_end_num <= axis_num_y else axis_num_y,
                    z_start_num if z_end_num <= axis_num_z else (
                            axis_num_z - size[2]): z_end_num if z_end_num <= axis_num_z else axis_num_z
                    ] = imgs[num, :, :, :]
                    num = num + 1
                    print(num)
        return output


if __name__ == '__main__':
    """
    -d:表示预测DICOM路径；
    -w:表示分割灰白质网络模型和权重路径；
    -g:表示预测结果输出路径;
    -n:表示预测的 nii 路径(可选参数)
    -l:表示用于计算 dice 的 label 路径, 格式为 nii (可选参数)
    """
    print('version: test 1.4.0')
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:w:g:n:l:", ["help"])
    except getopt.GetoptError as err:
        str_err = str(err)
        print('GetoptError:' + str_err)
        sys.exit(2)

    dicom_path = None
    model_path = None
    nii_path = None
    label_path = None
    save_path = None

    for opt, arg in opts:
        if opt == "-d":
            dicom_path = arg
            print('---dicom_path: ' + arg)
        elif opt == "-w":
            model_path = arg
            print('---model_path: ' + arg)
        elif opt == "-n":
            nii_path = arg
            print('---nii_path: ' + arg)
        elif opt == "-l":
            label_path = arg
            print('---label_path: ' + arg)
        elif opt == "-g":
            save_path = arg
            print('---save_path: ' + arg)
        else:
            print('input error: ' + 'unhandled option')
            assert False, "unhandled option"
    predict_obj = StaticSpacingPredict(mod_path=model_path,
                                       dicom_path=dicom_path,
                                       nii_path=nii_path,
                                       label_path=label_path,
                                       output_path=save_path,
                                       model_names=['MRI_brain_3DUet_seg.json', 'MRI_brain_seg_UNet3D.h5'])
    # predict_obj = MutabelSpacingPredict(mod_path=model_path,
    #                                     dicom_path=dicom_path,
    #                                     nii_path=nii_path,
    #                                     label_path=label_path,
    #                                     output_path=save_path,
    #                                     model_names=['MRI_brain_3DUet_seg.json', 'MRI_brain_seg_UNet3D.h5'])
    predict_obj.predict()
