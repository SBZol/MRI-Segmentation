"""
@time: 18-9-13
@author: zol
@contact: 13012215283@sina.cn
@file:  loss_function
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""

import numpy as np
from keras import backend as K


def dice_loss(cls, labels, predictions):
    arr = np.array([1. / 5639216, 1. / 1156998, 1. / 749851, 1. / 842545])
    weights = arr / np.sum(arr)
    print(weights)
    m, z, x, y, c = predictions.get_shape().as_list()
    total_loss = 0.
    for i in range(c):
        sub_pred = predictions[:, :, :, :, i]
        sub_label = labels[:, :, :, :, i]
        sub_loss = cls.normal_dice(sub_label, sub_pred)
        total_loss += sub_loss * weights[i]
    return 1 - total_loss


def dice_coef_4(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for 6 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    """
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f) + K.sum(y_pred_f)
    return K.mean((2. * intersect / (denom + smooth)))


def dice_coef_loss(y_true, y_pred):
    weight = [1, 1, 1, 1]
    total_dice = 0.
    for i in range(3):
        temp_true = y_true[..., i + 1]
        temp_pred = y_pred[..., i + 1]
        y_true_f = K.flatten(temp_true)
        y_pred_f = K.flatten(temp_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        temp_dice = (((2.0 * intersection) + 1e-10) / ((K.sum(y_true_f) + K.sum(y_pred_f)) + 1e-10))
        temp_dice *= weight[i + 1]
        total_dice += temp_dice

    total_dice /= 3
    return total_dice


def normal_dice(y_true, y_pred, smooth=1e-7):
    den = (K.sum(y_pred) + K.sum(y_true))
    dice = 2. * K.sum(y_pred * y_true) / (den + smooth)
    return dice


def veins_loss(y_true, y_pred):
    weight = [0.1, 1]
    total_loss = 0.
    for i in range(len(weight)):
        temp_true = y_true[..., i]
        temp_pred = y_pred[..., i]
        y_true_f = K.flatten(temp_true)
        y_pred_f = K.flatten(temp_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        temp_loss = 1 - (((2.0 * intersection) + 1e-10) / ((K.sum(y_true_f) + K.sum(y_pred_f)) + 1e-10))
        temp_loss *= weight[i]
        total_loss += temp_loss

    total_loss /= sum(weight)
    return total_loss

def veins_dice0(y_true, y_pred):
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersect + 1e-10) / (denom + 1e-10)

def veins_dice1(y_true, y_pred):
    y_true = y_true[..., 1]
    y_pred = y_pred[..., 1]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersect + 1e-10) / (denom + 1e-10)


def veins_dice2(y_true, y_pred):
    y_true = y_true[..., 2]
    y_pred = y_pred[..., 2]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersect + 1e-10) / (denom + 1e-10)


def veins_dice3(y_true, y_pred):
    y_true = y_true[..., 3]
    y_pred = y_pred[..., 3]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersect + 1e-10) / (denom + 1e-10)
