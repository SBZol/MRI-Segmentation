## 开发流程

开发**不使用 GitFlow**，而是使用 gitlab 官方推荐的 [Gitlab Flow](https://blog.csdn.net/jeffasd/article/details/49863665)。

## 目录结构

```plain
├─network
│  ├─unet_3d_nn.py
│  ├─unet_25d.py
│  ├─vnet3d_nn.py
│  ├─vnet25d.py
├─two_half_d (2.5D分割代码子目录)
│  ├─model
│  ├─create_train_data_25d.py
│  ├─generator_25d.py
│  ├─preprocess_25d.py
│  ├─train_25d.py
├─u_net_3d (3D U-Net 分割代码子目录)
│  ├─create_train_data.py
│  ├─generator.py
│  ├─postprocess.py
│  ├─predict.py
│  ├─train.py

├─__init__.py
├─ dataIO.py
├─ pathvariable.py
├─ lossfunction.py
├─ utils.py
├─ ...

```

## pip 包
可能会用到：
* pipprogressbar2
* keras==2.1.6
* tensorflow-gpu
* scikit-image
* nilearn
* nibabel
* SimpleItk
* opencv-contrib-python

