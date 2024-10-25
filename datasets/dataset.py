import os
import h5py
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as udata
import torchvision.transforms.functional as ttf


class pre_h5_dataset(udata.Dataset):
    def __init__(self, train=True):
        super(pre_h5_dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('Nuc_Project/data/h5_data/train.h5', 'r')
        else:
            h5f = h5py.File('Nuc_Project/data/h5_data/val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # ret data // keys->
        if self.train:
            h5f = h5py.File('Nuc_Project/data/h5_data/train.h5', 'r')
        else:
            h5f = h5py.File('Nuc_Project/data/h5_data/val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)


class real_nuc_dataset(udata.Dataset):
    def __init__(self, train=True):
        super(real_nuc_dataset, self).__init__()

    def __len__(self):
        return

    def __getitem__(self, index):
        return


class MyTrainDataSet(udata.Dataset):
    def __init__(self, inputPathTrain, transform):
        super(MyTrainDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)
        self.transform = transform

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath)
        # 图片转张量,并处理：
        if self.transform is not None:
            img = self.transform(inputImage)
        else:
            img = inputImage
        return img


class MyValueDataSet(udata.Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, transform):
        super(MyValueDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)
        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)
        self.transform = transform

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath)
        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath)
        # 图片转张量,并处理：
        if self.transform is not None:
            input_ = self.transform(inputImage)
            target = self.transform(targetImage)
        else:
            input_ = inputImage
            target = targetImage
        return input_, target


class MyTestDataSet(udata.Dataset):
    def __init__(self, inputPathTest):
        super(MyTestDataSet, self).__init__()

        self.inputPath = inputPathTest
        self.inputImages = os.listdir(inputPathTest)

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):
        index = index % len(self.inputImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        input_ = ttf.to_tensor(inputImage)

        return input_

