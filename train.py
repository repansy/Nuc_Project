import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.utils as utils
# from torchvision.models import ResNet
from models.DnCNN import DnCNN
from models.SwinIR import SwinIR
from models.FPNR import FPNR
from datasets.dataset import pre_h5_dataset
from utils.ValFunc import weights_init_kaiming, batch_PSNR, batch_SSIM
from utils.NoiseGenerator import NoiseGenerator
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')    # False
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")       # 128
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")     # 50
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="Nuc_Project/logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()


def DCNN_train():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = pre_h5_dataset(train=True)
    dataset_val = pre_h5_dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    # loader_value = DataLoader(dataset=dataset_val, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    criterion = nn.MSELoss(size_average=False)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    last_val = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            
            # noise 添加
            noise = NoiseGenerator(img_train, 25).stripe_noise(True, 'col', 0.85)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = noise.cuda() # noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            
            # results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            ssim_train = batch_SSIM(out_train, img_train, 1.)      
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                 (epoch+1, i+1, len(loader_train), loss.item(), psnr_train, ssim_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('SSIM on training data', ssim_train, step)
            step += 1
        #  the end of each epoch
        
        model.eval()
        # validate
        psnr_val = 0
        ssim_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            # noise = stripe_noise(img_val, 0, 30, 10)
            noise = NoiseGenerator(img_val, 25).stripe_noise(True, 'col', 0.85)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
            out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
            ssim_val += batch_SSIM(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        ssim_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f SSIM_val: %.4f" % (epoch + 1, psnr_val, ssim_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('SSIM on validation data', ssim_val, epoch)

        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        if psnr_val >= last_val:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))        
            last_val = psnr_val

def SWIR_train():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = pre_h5_dataset(train=True)
    dataset_val = pre_h5_dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    loader_value = DataLoader(dataset=dataset_val, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = SwinIR(img_size=40, patch_size=1, in_chans=1, window_size=8)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    criterion = nn.MSELoss(size_average=False)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    last_val = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            
            # noise 添加
            noise = NoiseGenerator(img_train, 25).stripe_noise(True, 'col', 0.85)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = noise.cuda() # noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            
            # results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            ssim_train = batch_SSIM(out_train, img_train, 1.)      
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                 (epoch+1, i+1, len(loader_train), loss.item(), psnr_train, ssim_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('SSIM on training data', ssim_train, step)
            step += 1
        #  the end of each epoch
        
        model.eval()
        # validate
        psnr_val = 0
        ssim_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            # noise = stripe_noise(img_val, 0, 30, 10)
            noise = NoiseGenerator(img_val, 25).stripe_noise(True, 'col', 0.85)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
            out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
            ssim_val += batch_SSIM(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        ssim_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f SSIM_val: %.4f" % (epoch + 1, psnr_val, ssim_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('SSIM on validation data', ssim_val, epoch)

        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        if psnr_val >= last_val:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))        
            last_val = psnr_val
        
        
def FPNR_train():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = pre_h5_dataset(train=True)
    dataset_val = pre_h5_dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    # loader_value = DataLoader(dataset=dataset_val, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = FPNR(channels=1, num_of_layers=5)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    criterion = nn.MSELoss(size_average=False)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    last_val = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            
            # noise 添加
            noise = NoiseGenerator(img_train, 25).stripe_noise(True, 'col', 0.85)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = noise.cuda() # noise = Variable(noise.cuda())
            _, _, out_train = model(imgn_train)
            loss = criterion(out_train, img_train) / (img_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            
            # results
            model.eval()
            _, _, out_train = model(imgn_train)
            # out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            ssim_train = batch_SSIM(out_train, img_train, 1.)      
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                 (epoch+1, i+1, len(loader_train), loss.item(), psnr_train, ssim_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('SSIM on training data', ssim_train, step)
            step += 1
        #  the end of each epoch
        
        model.eval()
        # validate
        psnr_val = 0
        ssim_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            # noise = stripe_noise(img_val, 0, 30, 10)
            noise = NoiseGenerator(img_val, 25).stripe_noise(True, 'col', 0.85)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
            _, _, out_val = model(imgn_val)
            # out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
            ssim_val += batch_SSIM(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        ssim_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f SSIM_val: %.4f" % (epoch + 1, psnr_val, ssim_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('SSIM on validation data', ssim_val, epoch)

        # log the images
        gain, offset, out_train = model(imgn_train)
        # out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        gain =  utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        offset = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        writer.add_image('gain', gain, epoch)
        writer.add_image('offset', offset, epoch)
        # save model
        if psnr_val >= last_val:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))        
            last_val = psnr_val
        
if __name__ == '__main__':
    # DCNN_train()     # batchsize:128
    FPNR_train()     # batchsize:64 
    # SWIR_train()   # batchsize:32
