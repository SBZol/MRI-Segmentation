# _*_ coding:utf-8 _*_
"""
@time: 2018/6/14
@author: zol
@contact: 13012215283@sina.cn
@file: pathvariable.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import os

# ROOT_PATH = r'D:\cliff\fcon_1000_anat' #cliff
ROOT_PATH = r'/media/chenzhuo/50FA53DAFA53BB44/DL/Data'

# zol
ORIGINAL_PATH = os.path.join(ROOT_PATH, 'train_data_sle')  # zol: data的原始路径

ROOT_PRE_PATH = os.path.join(ROOT_PATH, 'pre_data')  # zol: 统一管理处理后的训练数据

TRAIN_DATA_PATH = os.path.join(ROOT_PRE_PATH, 'reslice_train_data')  # zol：训练集数据存放路径

TRAIN_DATA_NII_PATH = os.path.join(ROOT_PRE_PATH, 'reslice_data_nii')  # zol：训练集数据存放路径

VAL_DATA_PATH = os.path.join(ROOT_PRE_PATH, 'validation_data')  # zol：验证集数据存放路径

TEST_DATA_PATH = os.path.join(ROOT_PRE_PATH, 'test_data')  # zol：测试集存放路径

PREDICT_PATH = os.path.join(ROOT_PRE_PATH, 'predict_data')  # zol：输出路径





def check_path():
    def check(path):
        if not os.path.isdir(path):
            os.makedirs(path)
            print('create dir: ' + path)

    check(ROOT_PRE_PATH)
    check(TRAIN_DATA_PATH)
    check(VAL_DATA_PATH)
    check(TEST_DATA_PATH)
    check(PREDICT_PATH)


if __name__ == '__main__':
    check_path()
