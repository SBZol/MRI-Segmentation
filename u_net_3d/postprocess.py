# _*_ coding:utf-8 _*_
"""
@time: 2018/6/13
@author: cliff
@contact: zshtom@163.com
@file: postprocess.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import os
import numpy as np
from nilearn import image
import cv2
import SimpleITK as sitk

from utils import np_2_nii, to_json, compress_nii


class Postprocess(object):
    def __init__(self, predict_nii, origin_nii, origin_nii_path, output_path):
        """
        构造方法
        Arguments:
            predict_nii -- 存放预测后nii对象
            original_nii -- 存放原始nii对象
        """
        self.predict_nii = predict_nii
        self.origin_nii = origin_nii
        self.origin_nii_path = origin_nii_path
        self.output_path = output_path

    def postprocess(self):

        res_nii_data = self.predict_nii.get_data()

        oringin_nii_data = self.origin_nii.get_data()

        print('------> getSpacing...')
        spacing = sitk.ReadImage(self.origin_nii_path).GetSpacing()

        # 计算intensity, volum
        print('------> calculate intensity and volume...')
        whtie_intensity, white_volume = self.calculate_intensity_volume(res_nii_data, oringin_nii_data, spacing, 1)
        gray_intensity, gray_volume = self.calculate_intensity_volume(res_nii_data, oringin_nii_data, spacing, 2)
        csf_intensity, csf_volume = self.calculate_intensity_volume(res_nii_data, oringin_nii_data, spacing, 3)

        # 生成intensity和volume的json
        gray_data = {'name': 'GrayMatter', 'intensityUnit': 'MRI_intensity', 'volumeUnit': 'mm^3',
                     'intensityValue': gray_intensity, 'volumeValue': gray_volume}
        white_data = {'name': 'WhiteMatter', 'intensityUnit': 'MRI_intensity', 'volumeUnit': 'mm^3',
                      'intensityValue': whtie_intensity, 'volumeValue': white_volume}
        csf_data = {'name': 'CSF', 'intensityUnit': 'MRI_intensity', 'volumeUnit': 'mm^3',
                    'intensityValue': csf_intensity, 'volumeValue': csf_volume}

        print('------> to json...')
        to_json([[csf_data, white_data, gray_data, ]], ['parts'],
                os.path.join(self.output_path, 'result.json'))

    @classmethod
    def calculate_intensity_volume(cls, predict_data, origin_data, spacing, label_num):
        """
         计算intensity 和 volume

         Arguments:
             predict_data -- 预测出的data
             origin_data -- 原始的data
             spacing -- spacing信息,例如（1.3, 1, 1）
             label_num -- 选定计算的标签 2:灰质, 3:白质, 4：脑脊液
         """
        predict_nii = predict_data.flatten()
        origin_vol = origin_data.flatten()

        # 计算intensity
        vol_arg = np.argwhere(predict_nii == label_num)
        label_vaule = origin_vol[vol_arg]
        intensity = np.mean(label_vaule, dtype=np.float64)

        # 计算volume
        volum = vol_arg.shape[0] * spacing[0] * spacing[1] * spacing[2]

        return intensity, volum

    @classmethod
    def edges_detection(cls, predict_nii, label_num, save_path):
        """
        对预测后的图片进行比例还原，并边缘检测

        Arguments:
            predict_nii -- 存放预测后nii对象
            label_num -- 选定分割内容
            save_path -- 处理完图像后保存的路径
        """
        r_nii_data = predict_nii.get_data()
        edges_data = []

        # edgesDetection
        for i in range(predict_nii.shape[0]):
            img_r_nii = r_nii_data[i, :, :]
            r_nii_data[r_nii_data != label_num] = 0
            img = img_r_nii.astype(np.uint8)
            dst = cv2.GaussianBlur(img, (3, 3), 0)
            edges = cv2.Canny(dst, 1, 4)
            edges_data.append(edges)

        edges_data = np.array(edges_data, dtype=np.float32)
        edges_data[edges_data == 255] = 1.0
        print(edges_data.shape)
        print(np.unique(edges_data))
        np_2_nii(predict_nii, edges_data, save_path)

        return edges_data

    # graph_cut
    # image close and image open


if __name__ == '__main__':
    pass
