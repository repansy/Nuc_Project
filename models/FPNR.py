import torch.nn as nn

class SCNAU(nn.Module):
    def __init__(self, channels, features, kernel_size, padding):
        super(SCNAU, self).__init__()
        


class CFConv(nn.Module):
    def __init__(self, channels, features, kernel_size, padding):
        super(CFConv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d())
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False, dilation=1))
        self.CFConv = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.CFConv(x)
        return out

        


class FPNR(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(FPNR, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.dncnn(x)
        return out
