import os
import h5py
import cv2
import glob
import numpy as np
from utils.Img2Patch import Im2Patch, normalize
from utils.DataAugmentation import data_augmentation


def prepare_h5_data(data_path, patch_size, stride, aug_times=1):

    # train_slice for large train data
    train_slice = 1
    for j in range(train_slice):
        # train
        print('process training data {}'.format(j))
        scales = [1, 0.9, 0.8, 0.7]
        # begin to change path
        files = glob.glob(os.path.join(data_path, 'train', '*.png'))
        files.sort()
        h5f = h5py.File('train_{}.h5'.format(j), 'w')
        train_num = 0
        for i in range(len(files)):

            img = cv2.imread(files[i])
            h, w, c = img.shape
            for k in range(len(scales)):
                # 改变大小并分块
                Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
                Img = np.expand_dims(Img[:, :, 0].copy(), 0)
                Img = np.float32(normalize(Img))
                patches = Im2Patch(Img, win=patch_size, stride=stride)
                # 展示scale
                print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
                for n in range(patches.shape[3]):
                    data = patches[:, :, :, n].copy()
                    h5f.create_dataset(str(train_num), data=data)
                    train_num += 1
                    for m in range(aug_times-1):
                        data_aug = data_augmentation(data, np.random.randint(1, 8))
                        h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                        train_num += 1
        h5f.close()
        # val
        print('\nprocess validation data')
        files.clear()
        files = glob.glob(os.path.join(data_path, 'val', '*.png'))
        files.sort()
        h5f = h5py.File('val_{}.h5'.format(j), 'w')
        val_num = 0
        for i in range(len(files)):
            print("file: %s" % files[i])
            img = cv2.imread(files[i])
            img = np.expand_dims(img[:, :, 0], 0)
            img = np.float32(normalize(img))
            h5f.create_dataset(str(val_num), data=img)
            val_num += 1
        h5f.close()
        print('training set, # samples %d\n' % train_num)
        print('val set, # samples %d\n' % val_num)


def prepare_data(data_path, patch_size, stride, aug_times=1):
    print(0)
    # 注意这里并不能随心所欲的使用上面的Im2Patch函数，应该每次将一个列的噪声取出来，尽量不破坏噪声的真实性
    
