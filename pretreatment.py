import os
import glob
import cv2
import h5py
import numpy as np
from datasets.prepare_data import prepare_h5_data

"""
将所有的预处理相关的工作，散布在各个文件里的部分整合然后输出
"""

        
if __name__ == '__main__':
    data_path = "/home/repansy/Nuc_Project/data/pre_data"
    patch_size = 40
    stride = 10
    '''
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path=data_path, patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path=data_path, patch_size=50, stride=10, aug_times=2)
    '''
    prepare_h5_data(data_path, patch_size, stride, aug_times=1)