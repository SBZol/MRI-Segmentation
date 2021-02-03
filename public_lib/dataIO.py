# _*_ coding:utf-8 _*_
"""
@time: 2018/6/8
@author: cliff
@contact: zshtom@163.com
@file: dataIO.py
@desc: Project of LinkingMed, Python3.6(64 bit), keras 2.1.6
"""
import os
import shutil
import gzip


def movefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件
        print("move %s -> %s" % (srcfile, dstfile))


def copyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


def unzip_gz(file_name):
    """ungz zip file"""

    # get the target file name (file_name without subfix of ".gz")
    unzip_name = file_name.replace(".gz", "")
    # create a gzip handle of that file
    g_file = gzip.GzipFile(file_name)
    # read the gzip data by read()，and then write the data down
    open(unzip_name, "wb").write(g_file.read())
    # close gzip handle
    g_file.close()


def compress_gz(fn_in, fn_out):
    """压缩文件"""
    f_in = open(fn_in, 'rb')
    f_out = gzip.open(fn_out, 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()


