import torch
import torch.nn as nn

class SCNAU(nn.Module):
    def __init__(self, channels):
        super(SCNAU, self).__init__()
        padding = 1
        branch_1 = []
        branch_1.append(nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=padding, bias=False))
        branch_1.append(nn.ReLU(inplace=True))
        branch_1.append(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=padding, bias=False))
        branch_1.append(nn.ReLU(inplace=True))
        branch_1.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=False)) # here no padding
        branch_1.append(nn.Sigmoid()) # 注意参数
        self.sp_branch = nn.Sequential(*branch_1)
        
        branch_2 = []
        branch_2.append(nn.AvgPool2d(1, stride=(1, 2)))  # 修改参数使得能够通过
        branch_2.append(nn.Linear(in_features = 20, out_features = 256, bias=False)) # 全连接
        branch_2.append(nn.ReLU(inplace=True))
        branch_2.append(nn.Linear(in_features = 256, out_features = 512, bias=False)) # 全连接
        branch_2.append(nn.ReLU(inplace=True))
        branch_2.append(nn.Linear(in_features = 512, out_features = 40, bias=False)) # 全连接
        branch_1.append(nn.Sigmoid())
        self.ch_branch = nn.Sequential(*branch_2)
    
    def forward(self, x):
        fs = self.sp_branch(x)
        fc = self.ch_branch(x)
        out = fs @ fc * x     # fc .* fs channel-wise * x @?
        return out
        
        


class CFConv(nn.Module):
    def __init__(self, channels):
        super(CFConv, self).__init__()
        kernel_size = 3
        padding = 1
        # std-conv
        fliter_1 = []
        fliter_1.append(nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=kernel_size, padding=padding, bias=False))
        fliter_1.append(nn.ReLU(inplace=True))
        self.std_fliter = nn.Sequential(*fliter_1)
        # sp-conv
        fliter_2 = []
        fliter_2.append(nn.MaxPool2d(kernel_size=2, stride=(2, 2)))
        fliter_2.append(nn.PixelShuffle(2))
        fliter_2.append(nn.ReLU(inplace=True))
        self.sp_fliter = nn.Sequential(*fliter_2)
        # dia-conv
        fliter_3 = []
        fliter_3.append(nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=kernel_size, padding=2, bias=False, dilation=2))
        fliter_3.append(nn.ReLU(inplace=True))
        self.dia_fliter = nn.Sequential(*fliter_3)
        # bottle_neck
        bottle_neck = []
        bottle_neck.append(nn.Conv2d(in_channels=112, out_channels=64, kernel_size=kernel_size, padding=padding, bias=False))
        bottle_neck.append(nn.ReLU(inplace=True))
        self.bottle_neck = nn.Sequential(*bottle_neck)
        
        # last 1*1*1
        
    def forward(self, x):
        # concatenated as a longchannel output vector
        fil_cat = torch.cat((self.sp_fliter(x), self.std_fliter(x), self.dia_fliter(x)), dim=1)
        # bottle neck
        out = self.bottle_neck(fil_cat)
        return out

        

class FEB_n(nn.Module):
    def __init__(self, channels):
        super(FEB_n, self).__init__()
        layers = []
        layers.append(CFConv(channels))
        layers.append(SCNAU(channels))
        self.FEB_n = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.FEB_n(x)
        return out
        

class FPNR(nn.Module):
    def __init__(self, channels, num_of_layers=5):
        super(FPNR, self).__init__()
        kernel_size = 3
        padding = 1
        self.sp_first = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=kernel_size, padding=padding, bias=False)
        gain = []
        offset = []
        for _ in range(num_of_layers):
            gain.append(FEB_n(64))
            offset.append(FEB_n(64)) # maybe dcnn or swinIR
        self.gain = nn.Sequential(*gain)
        self.offset = nn.Sequential(*offset)
        self.end = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, bias=False)
    
    def forward(self, x):
        x_sp = self.sp_first(x) 
        result_1 = self.gain(x_sp)
        gain = self.end(result_1) 
        
        next_x = self.sp_first(gain @ x)
        result_2 = self.offset(next_x)
        offset = self.end(result_2)
        out = gain + offset
        return gain, offset, out

if __name__ == "__main__":
    net = FPNR(1, 10)
    input = torch.randn(10, 1, 40, 40)
    a,b,c = net(input)
    print(a.size(), b.size(), c.size())