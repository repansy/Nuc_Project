import cv2
import os
import argparse
import glob
# import torch.fft
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from models import DnCNN
from utils import *
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default=r"G:\User\DnCNN-PyTorch-master\logs\stride_frequency\3", help='path of log files')
parser.add_argument("--test_data", type=str, default=r'infrared_data\original\val', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()


def normalize(data):
    return data/255.


def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('G:\\User\\DnCNN-PyTorch-master\\data', opt.test_data, '*.png'))[0:1]
    files_source.sort()

    # process data
    psnr_test = 0
    save_path = r'outputs_img'
    """
    # cv2在处理红外图像时，没有读到图片，初步猜测因为格式不对（RGBA）
    Img = cv2.imread(files_source[0], cv2.IMREAD_GRAYSCALE)
    image cv2_img = cv2.imread(f)
    """
    for f in files_source:
        pil_img = Image.open(f)
        cv2_img = np.asarray(pil_img)
        col = np.size(cv2_img, 1)
        Img = normalize(np.float32(cv2_img[:, :, 0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        noise = noise_generator(ISource, True, col)
        # noise = stripe_noise(ISource, 0, 20, 10)
        # noise = torch.FloatTensor(ISource.size()).normal_(mean=0.3, std=opt.test_noiseL/255.)
        # noise = torch.zeros_like(ISource)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad():  # this can save much memory
            print('have')
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        """
        # if you are using older version of PyTorch, torch.no_grad() may not be supported
        ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        """
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
        a = Out.data.cpu().numpy().astype(np.float32)
        img_1 = a[0, 0, :, :]
        cv2.imwrite(os.path.join(save_path, 'output_{}.png'.format(8)), img_1 * 255)
        psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)


if __name__ == "__main__":
    # 测试代码
    # main()
    # 进行对比
    path1 = r'G:\User\DnCNN-PyTorch-master\data\test_data\图4矫正后的图像.png'
    path2 = r'G:\User\DnCNN-PyTorch-master\outputs_img\output_7.png'
    # img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    pil_img = Image.open(path1)
    cv2_img = np.asarray(pil_img)
    img1 = cv2_img[:, :, 0]
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    diff = (img1-img2)*100
    cv2.imwrite("outputs_img/diff2.png", diff)
    cv2.imshow("diff", diff)
    cv2.waitKey()
    cv2.destroyAllWindows()
